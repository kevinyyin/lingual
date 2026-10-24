"""Probe gpt-realtime-translate: print every server event shape for one short clip.

    uv run python scripts/probe_openai.py [path/to/clip.wav] [--target es]

Streams the clip at 1x real time (20 ms chunks) plus 4 s of silence, sends
session.close, reads until session.closed, then prints each distinct event type with a
count and one example payload (base64 truncated). Saves results/probes/openai_<ts>.json.
Run this whenever the adapter (bench/providers/openai_translate.py) is in doubt.
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

URL = "wss://api.openai.com/v1/realtime/translations?model=gpt-realtime-translate"


def shorten(obj, n=80):
    if isinstance(obj, dict):
        return {k: shorten(v, n) for k, v in obj.items()}
    if isinstance(obj, list):
        return [shorten(v, n) for v in obj[:5]]
    if isinstance(obj, str) and len(obj) > n:
        return obj[:n] + f"...(+{len(obj) - n} chars)"
    return obj


async def main(clip: Path, target: str) -> None:
    load_dotenv(ROOT / ".env")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("OPENAI_API_KEY missing in latency-bench/.env")
    x, _ = load_wav(clip, 24000)
    x = np.concatenate([x, np.zeros(24000 * 4, np.float32)])
    pcm = to_pcm16(x)
    step = 24000 * 2 // 50
    chunks = [pcm[i:i + step] for i in range(0, len(pcm), step)]
    seen: dict[str, dict] = {}
    t0 = time.perf_counter_ns()

    async with websockets.connect(URL, additional_headers={"Authorization": f"Bearer {key}",
                                                           "OpenAI-Safety-Identifier": "latency-bench-probe"},
                                  max_size=None) as ws:
        async def reader():
            async for raw in ws:
                t = (time.perf_counter_ns() - t0) / 1e6
                ev = json.loads(raw)
                typ = ev.get("type")
                s = seen.setdefault(typ, {"count": 0, "first_ms": round(t, 1), "example": shorten(ev)})
                s["count"] += 1
                s["last_ms"] = round(t, 1)
                if typ in ("session.output_transcript.delta", "session.input_transcript.delta"):
                    print(f"{t:9.1f} ms {typ}: {ev.get('delta')!r}")
                elif typ != "session.output_audio.delta":
                    print(f"{t:9.1f} ms {typ}")
                if typ == "session.closed":
                    return

        rd = asyncio.create_task(reader())
        await asyncio.sleep(0.5)
        await ws.send(json.dumps({"type": "session.update", "session": {"audio": {
            "output": {"language": target}, "input": {"transcription": {"model": "gpt-realtime-whisper"}}}}}))
        await asyncio.sleep(0.5)
        start = time.perf_counter()
        for i, c in enumerate(chunks):
            await asyncio.sleep(max(0, start + i * 0.02 - time.perf_counter()))
            await ws.send(json.dumps({"type": "session.input_audio_buffer.append", "audio": base64.b64encode(c).decode()}))
        await ws.send(json.dumps({"type": "session.close"}))
        await asyncio.wait_for(rd, 30)

    print("\n=== event types ===")
    for typ, s in seen.items():
        print(f"{typ:40s} count={s['count']:5d} first={s['first_ms']} ms last={s.get('last_ms')} ms")
        print("   example:", json.dumps(s["example"], ensure_ascii=False)[:400])
    out = ROOT / "results" / "probes" / f"openai_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(seen, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", nargs="?", default=str(ROOT / "results" / "selftest" / "corpus" / "st_medium_1.wav"))
    ap.add_argument("--target", default="es")
    a = ap.parse_args()
    asyncio.run(main(Path(a.clip), a.target))
