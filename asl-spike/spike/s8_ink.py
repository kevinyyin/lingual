"""S8 (speech -> sign) stage: Cartesia Ink 2 turn detection latency.

The 30 barista utterances from S5 are synthesized once with Sonic (16 kHz PCM, voice Tessa), then each is
streamed to wss://api.cartesia.ai/stt/turns/websocket at 1x real time in 100 ms chunks on an absolute
schedule, followed by continuous digital silence (the docs say to send zeros, not gaps) until turn.end or
6 s. Measured from the end of audible speech (last sample above -45 dBFS, at its scheduled send time):
  eager_end_ms  arrival of turn.eager_end
  end_ms        arrival of turn.end (final transcript)
Writes results/s8_ink.jsonl and results/s8_ink_config.json. Sequential, one session per utterance.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import soundfile as sf
import websockets

sys.path.insert(0, os.path.dirname(__file__))
from s5_llm import E2G_ITEMS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SR = 16000
CHUNK_MS = 100
VOICE = "6ccbfb76-1fc6-48f7-b71d-91ac6298247b"  # Tessa
KEYTERMS = ["oat", "almond", "Maya", "decaf", "muffin", "latte"]
TTS_URL = "wss://api.cartesia.ai/tts/websocket?cartesia_version=2026-08-14"
STT_PARAMS = {"model": "ink-2", "encoding": "pcm_s16le", "sample_rate": SR, "cartesia_version": "2026-08-14"}


async def synth_all(key: str) -> list[np.ndarray]:
    out = []
    cache = ROOT / "cache/s8"
    cache.mkdir(parents=True, exist_ok=True)
    async with websockets.connect(TTS_URL, additional_headers={"X-API-Key": key}, max_size=None) as ws:
        for i, t in enumerate(E2G_ITEMS):
            p = cache / f"barista_{i:02d}.wav"
            if p.exists():
                out.append(sf.read(p, dtype="int16")[0])
                continue
            ctx = str(uuid.uuid4())
            await ws.send(json.dumps({"model_id": "sonic-3.6", "transcript": t, "voice": {"mode": "id", "id": VOICE},
                                      "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": SR},
                                      "language": "en", "context_id": ctx, "continue": False}))
            buf = []
            while True:
                r = json.loads(await ws.recv())
                if r.get("type") == "chunk" and r.get("data"):
                    buf.append(base64.b64decode(r["data"]))
                elif r.get("type") in ("done", "error"):
                    if r.get("type") == "error":
                        raise RuntimeError(str(r)[:300])
                    break
            pcm = np.frombuffer(b"".join(buf), np.int16)
            sf.write(p, pcm, SR)
            out.append(pcm)
    return out


async def stream_one(key: str, pcm: np.ndarray, text: str) -> dict:
    lead = np.zeros(int(0.5 * SR), np.int16)           # 0.5 s silence before speech
    audio = np.concatenate([lead, pcm])
    thr = int(32768 * 10 ** (-45 / 20))
    loud = np.where(np.abs(audio.astype(np.int32)) > thr)[0]
    speech_end_sample = int(loud[-1]) + 1 if len(loud) else len(audio)
    tail_max = int(6.0 * SR)
    audio = np.concatenate([audio, np.zeros(tail_max, np.int16)])
    n = int(SR * CHUNK_MS / 1000)
    q = urlencode(STT_PARAMS) + "".join(f"&keyterm={k}" for k in KEYTERMS)
    ev = []
    async with websockets.connect(f"wss://api.cartesia.ai/stt/turns/websocket?{q}",
                                  additional_headers={"X-API-Key": key}, max_size=None) as ws:
        done = asyncio.Event()

        async def recv():
            async for m in ws:
                if isinstance(m, bytes):
                    continue
                r = json.loads(m)
                ev.append((time.perf_counter_ns(), r))
                if r.get("type") in ("turn.end", "error"):
                    done.set()

        rt = asyncio.create_task(recv())
        t0 = time.perf_counter_ns()
        for k in range(0, len(audio), n):
            target = t0 + int(k / SR * 1e9)
            while (d := target - time.perf_counter_ns()) > 0:
                await asyncio.sleep(min(d / 1e9, 0.01))
            await ws.send(audio[k:k + n].tobytes())
            if done.is_set():
                break
        await ws.send(json.dumps({"type": "close"}))
        try:
            await asyncio.wait_for(rt, timeout=10)
        except asyncio.TimeoutError:
            rt.cancel()
    t_end = t0 + int(speech_end_sample / SR * 1e9)

    def first(tp):
        return next(((t, r) for t, r in ev if r.get("type") == tp), (None, None))
    te, re_ = first("turn.eager_end")
    tn, rn = first("turn.end")
    err = next((r for _, r in ev if r.get("type") == "error"), None)
    return {"text": text, "speech_s": round(len(pcm) / SR, 3), "eager_end_ms": (te - t_end) / 1e6 if te else None,
            "end_ms": (tn - t_end) / 1e6 if tn else None, "transcript": (rn or {}).get("transcript"),
            "eager_transcript": (re_ or {}).get("transcript"), "resumes": sum(1 for _, r in ev if r.get("type") == "turn.resume"),
            "n_turn_end": sum(1 for _, r in ev if r.get("type") == "turn.end"), "error": err}


async def main():
    key = os.environ.get("CARTESIA_API_KEY")
    if not key:
        print("skipped: no key")
        return
    json.dump({"stt": STT_PARAMS, "keyterms": KEYTERMS, "chunk_ms": CHUNK_MS, "tts_voice": VOICE,
               "speech_end": "last sample above -45 dBFS at its scheduled send time", "lead_silence_s": 0.5},
              open(ROOT / "results/s8_ink_config.json", "w"), indent=1)
    pcms = await synth_all(key)
    with open(ROOT / "results/s8_ink.jsonl", "w", encoding="utf-8") as f:
        for i, (p, t) in enumerate(zip(pcms, E2G_ITEMS)):
            try:
                r = await stream_one(key, p, t)
            except Exception as e:
                r = {"text": t, "error": f"{type(e).__name__}: {e}"[:300]}
            r["i"] = i
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(i, r.get("end_ms"), r.get("eager_end_ms"), "|", r.get("transcript"), "|", r.get("error"), flush=True)
            if r.get("error") and any(w in str(r["error"]).lower() for w in ("402", "401", "403", "429", "credit", "quota")):
                print("API FAILURE, stopping")
                sys.exit(2)


if __name__ == "__main__":
    asyncio.run(main())
