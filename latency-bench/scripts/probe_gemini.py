"""Probe Gemini Live Translate: print every server message shape for one short clip.

    uv run python scripts/probe_gemini.py [path/to/clip.wav] [--target es] [--model gemini-3.5-live-translate-preview]

Sends the setup message from the Live Translate docs, waits for setupComplete, streams the
clip at 1x real time (20 ms chunks, 16 kHz) plus 4 s of silence, sends audioStreamEnd and
reads until 5 s pass with no message. Prints each distinct message shape (key path) with a
count and an example (base64 truncated). Saves results/probes/gemini_<ts>.json.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import websockets  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from bench.audio import load_wav, to_pcm16  # noqa: E402

URL = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"


def shorten(obj, n=80):
    if isinstance(obj, dict):
        return {k: shorten(v, n) for k, v in obj.items()}
    if isinstance(obj, list):
        return [shorten(v, n) for v in obj[:5]]
    if isinstance(obj, str) and len(obj) > n:
        return obj[:n] + f"...(+{len(obj) - n} chars)"
    return obj


def shape(obj, prefix=""):
    """Key paths present in a message, e.g. serverContent.modelTurn.parts[].inlineData.data"""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += shape(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list) and obj:
        out += shape(obj[0], prefix + "[]")
    else:
        out.append(prefix)
    return out


async def main(clip: Path, target: str, model: str) -> None:
    load_dotenv(ROOT / ".env")
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY missing in latency-bench/.env")
    x, _ = load_wav(clip, 16000)
    pcm = to_pcm16(np.concatenate([x, np.zeros(16000 * 4, np.float32)]))
    step = 16000 * 2 // 50
    chunks = [pcm[i:i + step] for i in range(0, len(pcm), step)]
    seen: dict[str, dict] = {}
    t0 = time.perf_counter_ns()
    last = [time.perf_counter()]

    async with websockets.connect(f"{URL}?key={key}", max_size=None) as ws:
        # API reference: transcription configs are top-level setup fields; translationConfig is
        # inside generationConfig (the Live Translate guide's raw-WebSocket example nests both)
        await ws.send(json.dumps({"setup": {"model": f"models/{model}",
            "inputAudioTranscription": {}, "outputAudioTranscription": {},
            "generationConfig": {"responseModalities": ["AUDIO"],
                                 "translationConfig": {"targetLanguageCode": target, "echoTargetLanguage": False}}}}))
        ready = asyncio.Event()

        async def reader():
            try:
                async for raw in ws:
                    if b'"text"' in (raw if isinstance(raw, bytes) else raw.encode()) or b'inlineData' in (raw if isinstance(raw, bytes) else raw.encode()):
                        last[0] = time.perf_counter()  # empty languageCode-only messages do not count as activity
                    t = (time.perf_counter_ns() - t0) / 1e6
                    msg = json.loads(raw)
                    sig = " | ".join(sorted(set(shape(msg))))
                    s = seen.setdefault(sig, {"count": 0, "first_ms": round(t, 1), "example": shorten(msg)})
                    s["count"] += 1
                    if "setupComplete" in msg:
                        ready.set()
                    sc = msg.get("serverContent", {})
                    for k in ("inputTranscription", "outputTranscription"):
                        if k in sc:
                            print(f"{t:9.1f} ms {k}: {sc[k]}")
                    for k in ("turnComplete", "generationComplete", "interrupted"):
                        if sc.get(k):
                            print(f"{t:9.1f} ms {k}")
            except websockets.ConnectionClosed as e:
                print("connection closed:", e)

        rd = asyncio.create_task(reader())
        await asyncio.wait_for(ready.wait(), 20)
        print("setupComplete")
        start = time.perf_counter()
        for i, c in enumerate(chunks):
            await asyncio.sleep(max(0, start + i * 0.02 - time.perf_counter()))
            await ws.send(json.dumps({"realtimeInput": {"audio": {"data": base64.b64encode(c).decode(),
                                                                  "mimeType": "audio/pcm;rate=16000"}}}))
        await ws.send(json.dumps({"realtimeInput": {"audioStreamEnd": True}}))
        t_end = time.perf_counter()
        while time.perf_counter() - last[0] < 5 and time.perf_counter() - t_end < 15 and not rd.done():
            await asyncio.sleep(0.2)
        rd.cancel()

    print("\n=== message shapes ===")
    for sig, s in seen.items():
        print(f"count={s['count']:5d} first={s['first_ms']} ms :: {sig}")
        print("   example:", json.dumps(s["example"], ensure_ascii=False)[:400])
    out = ROOT / "results" / "probes" / f"gemini_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(seen, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", nargs="?", default=str(ROOT / "results" / "selftest" / "corpus" / "st_medium_1.wav"))
    ap.add_argument("--target", default="es")
    ap.add_argument("--model", default="gemini-3.5-live-translate-preview")
    a = ap.parse_args()
    asyncio.run(main(Path(a.clip), a.target, a.model))
