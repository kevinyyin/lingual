"""Adapters vs. local fake servers that speak the documented wire protocols.

These do NOT prove the real APIs behave this way (that is what scripts/probe_*.py are
for); they prove each adapter handles the documented handshake, streaming, close/drain
and event mapping end to end through the real runner.
"""

from __future__ import annotations

import asyncio
import base64
import json

import numpy as np
import pytest
from websockets.asyncio.server import serve

from bench import clock
from bench.analysis.metrics import cascade_stages, latency_metrics, output_chunks
from bench.audio import room_tone, save_wav, to_pcm16
from bench.config import BenchConfig, LLMCfg
from bench.corpus import PreparedClip
from bench.events import SourceTiming, read_run
from bench.runner import Job, RunPaths, run_one

SR = 24000


def tone(n: int, sr: int = 24000) -> bytes:
    return to_pcm16(0.3 * np.sin(2 * np.pi * 220 * np.arange(n) / sr))


def burst_clip(tmp_path) -> PreparedClip:
    x = room_tone(int(4.0 * SR), SR, -60.0)
    x[SR: SR + int(1.2 * SR)] += 0.4 * np.sin(2 * np.pi * 300 * np.arange(int(1.2 * SR)) / SR)
    p = tmp_path / "clip.wav"
    save_wav(p, x, SR)
    return PreparedClip(id="clip", pair="es-en", category="medium", source_lang="es", synthetic=True, file=str(p),
                        sample_rate=SR, duration_s=4.0, sha256="x",
                        timing=SourceTiming(sample_rate=SR, n_samples=len(x), onset_sample=SR,
                                            offset_sample=SR + int(1.2 * SR)))


# ------------------------------------------------------------------ OpenAI
async def fake_openai(ws):
    await ws.send(json.dumps({"type": "session.created", "event_id": "e0", "session": {"id": "sess_1", "type": "translation"}}))
    received = 0
    emitted = 0
    async for raw in ws:
        ev = json.loads(raw)
        if ev["type"] == "session.update":
            assert ev["session"]["audio"]["output"]["language"] == "en"
            await ws.send(json.dumps({"type": "session.updated", "event_id": "e1", "session": ev["session"]}))
        elif ev["type"] == "session.input_audio_buffer.append":
            received += len(base64.b64decode(ev["audio"])) // 2
            # one 200 ms engine frame of "translation" per 200 ms of input, after 1 s of input
            while received >= 24000 + (emitted + 1) * 4800 and emitted < 6:
                emitted += 1
                await ws.send(json.dumps({"type": "session.output_audio.delta", "event_id": f"a{emitted}",
                                          "delta": base64.b64encode(tone(4800)).decode(), "sample_rate": 24000,
                                          "channels": 1, "format": "pcm16", "elapsed_ms": emitted * 200}))
                await ws.send(json.dumps({"type": "session.output_transcript.delta", "delta": " word", "elapsed_ms": 0}))
                await ws.send(json.dumps({"type": "session.input_transcript.delta", "delta": " palabra"}))
        elif ev["type"] == "session.close":
            await ws.send(json.dumps({"type": "session.output_audio.delta", "delta": base64.b64encode(tone(2400)).decode()}))
            await ws.send(json.dumps({"type": "session.closed", "event_id": "ez"}))


def test_openai_adapter_against_fake_server(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    from bench.providers.openai_translate import OpenAITranslateProvider

    async def go():
        async with serve(fake_openai, "127.0.0.1", 0) as srv:
            port = srv.sockets[0].getsockname()[1]
            prov = OpenAITranslateProvider({"url": f"ws://127.0.0.1:{port}/v1/realtime/translations",
                                            "input_transcription_model": "gpt-realtime-whisper", "input_sample_rate": 24000})
            cfg = BenchConfig()
            paths = RunPaths(tmp_path / "raw", tmp_path / "audio")
            return await run_one(prov, Job("openai", burst_clip(tmp_path), 0), cfg, "t", paths, {}, {})

    raw = clock.run(go())
    run = read_run(raw)
    assert run.footer.status == "ok", run.footer.error
    kinds = [e.kind for e in run.events]
    assert kinds.count("audio_out") == 7
    assert "text_out_delta" in kinds and "source_text_delta" in kinds
    assert any(e.stage == "openai.session.closed" for e in run.events)
    assert run.connect.t_session_ready_ns > run.connect.t_connect_start_ns
    assert "".join(e.text for e in run.events if e.kind == "text_out_delta").strip().startswith("word")


# ------------------------------------------------------------------ Gemini
async def fake_gemini(ws):
    setup = json.loads(await ws.recv())
    assert setup["setup"]["generationConfig"]["translationConfig"]["targetLanguageCode"] == "en"
    await ws.send(json.dumps({"setupComplete": {}}).encode())
    received = 0
    emitted = 0
    async for raw in ws:
        msg = json.loads(raw)
        ri = msg.get("realtimeInput", {})
        if "audio" in ri:
            assert ri["audio"]["mimeType"] == "audio/pcm;rate=16000"
            received += len(base64.b64decode(ri["audio"]["data"])) // 2
            while received >= 16000 + (emitted + 1) * 1600 and emitted < 10:
                emitted += 1
                await ws.send(json.dumps({"serverContent": {
                    "modelTurn": {"parts": [{"inlineData": {"mimeType": "audio/pcm;rate=24000",
                                                            "data": base64.b64encode(tone(2400)).decode()}}]},
                    "outputTranscription": {"text": "hi ", "languageCode": "en"}}}).encode())
        if ri.get("audioStreamEnd"):
            await ws.send(json.dumps({"serverContent": {"turnComplete": True}}).encode())


def test_gemini_adapter_against_fake_server(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    from bench.providers.gemini_translate import GeminiTranslateProvider

    async def go():
        async with serve(fake_gemini, "127.0.0.1", 0) as srv:
            port = srv.sockets[0].getsockname()[1]
            prov = GeminiTranslateProvider({"url": f"ws://127.0.0.1:{port}/ws", "drain_idle_s": 0.5,
                                            "drain_max_s": 5, "language_codes": {"zh": "zh-Hans"}})
            return await run_one(prov, Job("gemini", burst_clip(tmp_path), 0), BenchConfig(), "t",
                                 RunPaths(tmp_path / "raw", tmp_path / "audio"), {}, {})

    raw = clock.run(go())
    run = read_run(raw)
    assert run.footer.status == "ok", run.footer.error
    assert sum(e.kind == "audio_out" for e in run.events) == 10
    assert any(e.stage == "gemini.turnComplete" for e in run.events)
    assert any(e.stage == "gemini.drained" for e in run.events)


# ------------------------------------------------------------------ Cascade: Flux + LLM + Cartesia
async def fake_flux(ws):
    await ws.send(json.dumps({"type": "Connected", "request_id": "r", "sequence_id": 0}))
    received, turn_started, ended = 0, False, False
    words = ["hola", "mundo", "como", "estas", "hoy"]
    last_update = 0
    async for raw in ws:
        if isinstance(raw, bytes):
            received += len(raw) // 2
            t = received / 16000
            if t > 1.2 and not turn_started:
                turn_started = True
                await ws.send(json.dumps({"type": "TurnInfo", "event": "StartOfTurn", "turn_index": 0,
                                          "transcript": "hola", "words": [{"word": "hola"}], "languages": ["es"]}))
            if turn_started and not ended and t - last_update >= 0.25 and t < 2.3:
                last_update = t
                n = min(len(words), 1 + int((t - 1.2) / 0.25))
                await ws.send(json.dumps({"type": "TurnInfo", "event": "Update", "turn_index": 0,
                                          "transcript": " ".join(words[:n]), "words": [{"word": w} for w in words[:n]]}))
            if t > 2.4 and not ended:
                ended = True
                tr = " ".join(words)
                await ws.send(json.dumps({"type": "TurnInfo", "event": "EagerEndOfTurn", "turn_index": 0,
                                          "transcript": tr, "words": [{"word": w} for w in words]}))
                await asyncio.sleep(0.15)
                await ws.send(json.dumps({"type": "TurnInfo", "event": "EndOfTurn", "turn_index": 0, "transcript": tr,
                                          "words": [{"word": w} for w in words], "trigger": "model", "languages": ["es"]}))
        else:
            msg = json.loads(raw)
            if msg["type"] == "CloseStream":
                await ws.close()
                return


CARTESIA_LOG: list[dict] = []


async def fake_cartesia(ws):
    """Speaks each transcript as one 100 ms chunk. Chunks carry flush_id (starts at 1,
    incremented by each flush, as documented); continue=false ends the context with done."""
    flush_ids: dict[str, int] = {}
    async for raw in ws:
        msg = json.loads(raw)
        CARTESIA_LOG.append(msg)
        assert msg["output_format"]["encoding"] == "pcm_s16le"
        ctx = msg["context_id"]
        fid = flush_ids.setdefault(ctx, 1)
        if msg["transcript"]:
            await asyncio.sleep(0.08)
            await ws.send(json.dumps({"type": "chunk", "context_id": ctx, "done": False, "status_code": 206,
                                      "flush_id": fid, "data": base64.b64encode(tone(2400)).decode()}))
        if msg.get("flush"):
            await ws.send(json.dumps({"type": "flush_done", "context_id": ctx, "flush_done": True,
                                      "flush_id": fid, "done": False, "status_code": 206}))
            flush_ids[ctx] = fid + 1
        if not msg["continue"]:
            await ws.send(json.dumps({"type": "done", "context_id": ctx, "done": True, "status_code": 206}))


async def fake_llm_http(reader, writer):
    req = await reader.readuntil(b"\r\n\r\n")
    head = req.decode()
    length = int(next((ln.split(":")[1] for ln in head.split("\r\n") if ln.lower().startswith("content-length")), "0"))
    body = await reader.readexactly(length) if length else b""
    if head.startswith("GET"):
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}")
    else:
        assert json.loads(body)["stream"] is True
        await asyncio.sleep(0.1)
        chunks = "".join(f"data: {json.dumps({'choices': [{'delta': {'content': t}}]})}\n\n"
                         for t in ["Hello ", "world, ", "how are ", "you today?"]) + "data: [DONE]\n\n"
        data = chunks.encode()
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: "
                     + str(len(data)).encode() + b"\r\n\r\n" + data)
    await writer.drain()
    writer.close()


VARIANTS = {
    "on_turn_end": {"translate_mode": "on_turn_end"},
    "incremental": {"translate_mode": "incremental"},
    "inc-per_turn": {"translate_mode": "incremental", "tts_context": "per_turn"},
    "inc-per_turn-flush": {"translate_mode": "incremental", "tts_context": "per_turn", "tts_flush_each": True},
    "inc-per_turn-concise-adaptive": {"translate_mode": "incremental", "tts_context": "per_turn",
                                      "prompt_variant": "concise", "adaptive_speed": {"enabled": True}},
    "inc-per_turn-pipelined": {"translate_mode": "incremental", "tts_context": "per_turn",
                               "pipeline_fragments": True},
}


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_cascade_adapter_against_fake_servers(tmp_path, monkeypatch, variant):
    mode = VARIANTS[variant]["translate_mode"]
    CARTESIA_LOG.clear()
    for k in ("DEEPGRAM_API_KEY", "CARTESIA_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(k, "test")
    from bench.providers.cascade import CascadeProvider

    async def go():
        llm_srv = await asyncio.start_server(fake_llm_http, "127.0.0.1", 0)
        llm_port = llm_srv.sockets[0].getsockname()[1]
        async with serve(fake_flux, "127.0.0.1", 0) as dg, serve(fake_cartesia, "127.0.0.1", 0) as ct:
            bench = BenchConfig(llm={"translate": LLMCfg(base_url=f"http://127.0.0.1:{llm_port}/v1", model="fake")})
            cfg = {"_bench": bench, **VARIANTS[variant], "use_eager_eot": True, "drain_max_s": 10,
                   "incremental": {"stable_updates": 2, "holdback_words": 1, "min_segment_words": 2},
                   "stt": {"url": f"ws://127.0.0.1:{dg.sockets[0].getsockname()[1]}/v2/listen",
                           "supported_languages": ["en", "es"], "eager_eot_threshold": 0.5},
                   "tts": {"url": f"ws://127.0.0.1:{ct.sockets[0].getsockname()[1]}/tts/websocket",
                           "default_voice_id": "v"}}
            prov = CascadeProvider(cfg)
            raw = await run_one(prov, Job("cascade", burst_clip(tmp_path), 0), bench, "t",
                                RunPaths(tmp_path / "raw", tmp_path / "audio"), {}, {})
        llm_srv.close()
        return raw

    raw = clock.run(go())
    run = read_run(raw)
    assert run.footer.status == "ok", run.footer.error
    stages = [e.stage for e in run.events if e.kind == "stage_mark"]
    for s in ("stt.ready", "stt.end_of_turn", "llm.request", "llm.first_token", "llm.done",
              "tts.first_text", "tts.first_audio", "tts.done"):
        assert s in stages, (s, stages)
    import soundfile as sf

    pcm = sf.read(str(tmp_path / "audio" / run.footer.output_audio_file), dtype="int16")[0]
    chunks = output_chunks(run, pcm, -45.0)
    m = latency_metrics(run, chunks)
    d = cascade_stages(run, chunks)
    assert m["end_lag_arrival_ms"] is not None
    if mode == "on_turn_end":
        assert "llm.speculative_hit" in stages
        parts = sum(d[k] for k in ("stt_finalize_ms", "translate_wait_ms", "tts_first_byte_ms", "tts_stream_rest_ms"))
        assert parts == pytest.approx(m["end_lag_arrival_ms"], abs=0.01)
    else:
        assert stages.count("llm.request") >= 2  # at least one stable-prefix segment + the tail
        segs = [e.meta.get("seg") for e in run.events if e.kind == "audio_out"]
        assert segs == sorted(segs)  # sequencer keeps segment audio in order
    contexts = {m["context_id"] for m in CARTESIA_LOG}
    if VARIANTS[variant].get("tts_context") == "per_turn":
        assert contexts == {"turn-0"}  # one synthesis context for the whole turn
        assert [m["continue"] for m in CARTESIA_LOG if not m.get("flush")][-1] is False  # tail closes it
        assert sum(1 for m in CARTESIA_LOG if not m["continue"]) == 1
        if VARIANTS[variant].get("tts_flush_each"):
            assert any(m.get("flush") for m in CARTESIA_LOG)
            assert any(e.meta.get("attributed") == "flush_id" for e in run.events if e.stage == "tts.first_audio")
    else:
        assert len(contexts) == stages.count("llm.request") - stages.count("llm.cancelled")
    if VARIANTS[variant].get("adaptive_speed"):
        assert all("generation_config" in m for m in CARTESIA_LOG)
    if VARIANTS[variant].get("pipeline_fragments"):
        reqs = [e for e in run.events if e.stage == "llm.request"]
        assert any(e.meta.get("pipelined") for e in reqs)


def test_cascade_unsupported_pair_is_recorded_not_failed(tmp_path, monkeypatch):
    from bench.providers.cascade import CascadeProvider

    bench = BenchConfig(llm={"translate": LLMCfg()})
    prov = CascadeProvider({"_bench": bench, "stt": {"supported_languages": ["en", "es"]}})
    clip = burst_clip(tmp_path).model_copy(update={"pair": "zh-en"})
    raw = clock.run(run_one(prov, Job("cascade", clip, 0), bench, "t",
                            RunPaths(tmp_path / "raw", tmp_path / "audio"), {}, {}))
    assert read_run(raw).footer.status == "unsupported"
