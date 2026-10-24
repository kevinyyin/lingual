"""Session rotation (flush on pause / watchdog) against a fake translation server that
holds ALL translated audio until session.close, the worst case round 1 observed."""

from __future__ import annotations

import asyncio
import base64
import json

import numpy as np
import pytest
import soundfile as sf
from websockets.asyncio.server import serve

from bench import clock
from bench.analysis.metrics import PlaybackPolicy, output_chunks, simulate_playback_policy, utterance_metrics
from bench.audio import room_tone, save_wav, to_pcm16
from bench.causal_vad import VadEvent
from bench.config import BenchConfig
from bench.corpus import PreparedClip
from bench.events import SourceTiming, read_run
from bench.runner import Job, RunPaths, run_one

SR = 24000
OPENED: list[int] = []


def tone(n: int) -> bytes:
    return to_pcm16(0.3 * np.sin(2 * np.pi * 220 * np.arange(n) / SR))


async def holding_server(ws):
    """Translation server that emits nothing until session.close, then flushes."""
    OPENED.append(1)
    await ws.send(json.dumps({"type": "session.created", "session": {"id": f"sess_{len(OPENED)}"}}))
    received = 0
    async for raw in ws:
        ev = json.loads(raw)
        if ev["type"] == "session.update":
            await ws.send(json.dumps({"type": "session.updated", "session": ev["session"]}))
        elif ev["type"] == "session.input_audio_buffer.append":
            a = np.frombuffer(base64.b64decode(ev["audio"]), "<i2")
            received += int(np.sum(np.abs(a.astype(np.int32)) > 3000))  # only loud samples ("speech")
        elif ev["type"] == "session.close":
            await asyncio.sleep(0.15)  # processing
            # "translate" only if it heard speech: a fixed 1 s of audio
            if received > SR // 4:
                for _ in range(5):
                    await ws.send(json.dumps({"type": "session.output_audio.delta",
                                              "delta": base64.b64encode(tone(4800)).decode(), "sample_rate": 24000}))
            await ws.send(json.dumps({"type": "session.closed"}))
            return


class ScriptedVAD:
    """speech_end at fixed chunk indices (stands in for the causal VAD)."""

    def __init__(self, end_chunks: list[int]):
        self.end_chunks = set(end_chunks)
        self.i = 0

    def push(self, pcm: bytes, t_ns: int):
        i = self.i
        self.i += 1
        return [VadEvent("speech_end", t_ns, i * 320)] if i in self.end_chunks else []


def two_utterance_clip(tmp_path) -> PreparedClip:
    # room tone 1 s | burst 1.5 s | pause 1.5 s | burst 1.5 s | tail 2 s  (total 7.5 s)
    x = room_tone(int(7.5 * SR), SR, -60.0)
    t = np.arange(int(1.5 * SR)) / SR
    for start in (1.0, 4.0):
        a = int(start * SR)
        x[a: a + len(t)] += 0.4 * np.sin(2 * np.pi * 300 * t)
    p = tmp_path / "two.wav"
    save_wav(p, x, SR)
    segs = [(SR, int(2.5 * SR)), (4 * SR, int(5.5 * SR))]
    return PreparedClip(id="two", pair="es-en", category="two_utterances", source_lang="es", synthetic=True,
                        file=str(p), sample_rate=SR, duration_s=7.5, sha256="x",
                        timing=SourceTiming(sample_rate=SR, n_samples=len(x), onset_sample=SR,
                                            offset_sample=int(5.5 * SR), segments=segs))


def run_policy(tmp_path, monkeypatch, policy: str, vad_ends: list[int], **extra):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    OPENED.clear()
    from bench.providers.openai_translate import OpenAITranslateProvider

    async def go():
        async with serve(holding_server, "127.0.0.1", 0) as srv:
            port = srv.sockets[0].getsockname()[1]
            cfg = {"url": f"ws://127.0.0.1:{port}/v1/realtime/translations", "flush_policy": policy,
                   "_vad": ScriptedVAD(vad_ends) if policy != "none" else None, "close_timeout_s": 5, **extra}
            prov = OpenAITranslateProvider(cfg)
            return await run_one(prov, Job("openai", two_utterance_clip(tmp_path), 0), BenchConfig(), "t",
                                 RunPaths(tmp_path / "raw", tmp_path / "audio"), {}, {})

    run = read_run(clock.run(go()))
    assert run.footer.status == "ok", run.footer.error
    pcm = sf.read(str(tmp_path / "audio" / run.footer.output_audio_file), dtype="int16")[0]
    chunks = output_chunks(run, pcm, -45.0)
    utts = utterance_metrics(run, chunks, simulate_playback_policy(chunks, PlaybackPolicy("none")))
    return run, utts


def test_no_flush_policy_holds_everything_until_finish(tmp_path, monkeypatch):
    run, utts = run_policy(tmp_path, monkeypatch, "none", [])
    assert len(OPENED) == 1
    # all audio arrives after the stream ends: utterance 0 gets nothing, utterance 1 ends > 2 s late
    assert utts[0]["n_chunks"] == 0
    assert utts[1]["end_lag_playback_ms"] > 2000


def test_on_pause_rotation_flushes_each_utterance(tmp_path, monkeypatch):
    # speech_end ~0.5 s after each burst: chunk (2.5 + 0.5) / 0.02 = 150 and (5.5 + 0.5) / 0.02 = 300
    run, utts = run_policy(tmp_path, monkeypatch, "on_pause", [150, 300])
    stages = [e.stage for e in run.events if e.kind == "stage_mark"]
    assert stages.count("rotation.trigger") == 2 and stages.count("rotation.promote") == 2
    assert stages.count("cost.session") == len(OPENED) >= 4  # initial + 3 standbys (last one unused)
    # the first session's audio arrives after promotion and is marked as a drain
    drained = [e for e in run.events if e.kind == "audio_out" and e.meta.get("after_promotion")]
    assert drained and all(e.meta["session"] in (0, 1) for e in drained)
    # each utterance is voiced ~0.5 s (VAD) + 0.15 s (server) after its offset, + 1 s of audio
    for u in utts:
        assert u["n_chunks"] > 0
        assert 1400 < u["end_lag_playback_ms"] < 2200, utts
    # standby sessions received no audio until promoted; the unused last standby got none
    sess = {e.meta["session"]: e.meta for e in run.events if e.stage == "cost.session"}
    assert sess[max(sess)]["audio_s"] == 0


def test_watchdog_fires_only_without_audible_audio(tmp_path, monkeypatch):
    run, utts = run_policy(tmp_path, monkeypatch, "watchdog", [150, 300], watchdog_ms=400)
    stages = [e.stage for e in run.events if e.kind == "stage_mark"]
    # the holding server never speaks on its own, so both watchdogs fire
    assert stages.count("watchdog.arm") == 2
    assert stages.count("rotation.trigger") == 2
    reasons = [e.meta.get("reason") for e in run.events if e.stage == "rotation.trigger"]
    assert reasons == ["watchdog", "watchdog"]
    for u in utts:
        assert u["n_chunks"] > 0 and u["end_lag_playback_ms"] < 2600
