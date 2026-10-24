"""Probe: does an idle, configured translation session need keepalive audio?

Opens a session, sends session.update, then sends NO audio for --idle seconds while
logging any server events (errors, closes). Then streams a clip at 1x real time, sends
session.close and reports whether translation still arrives and how quickly.

    uv run python scripts/probe_openai_idle.py [--idle 90] [clip.wav]
Writes results/probes/openai_idle_<ts>.json.
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


async def main(clip: Path, idle: float) -> None:
    load_dotenv(ROOT / ".env")
    key = os.environ["OPENAI_API_KEY"]
    t0 = time.perf_counter()
    log: list[dict] = []
    first_audible = None
    closed = asyncio.Event()

    def rel() -> float:
        return round(time.perf_counter() - t0, 3)

    async with websockets.connect(URL, additional_headers={"Authorization": f"Bearer {key}",
                                                           "OpenAI-Safety-Identifier": "latency-bench-probe"},
                                  max_size=None, ping_interval=20) as ws:
        async def reader():
            nonlocal first_audible
            try:
                async for raw in ws:
                    ev = json.loads(raw)
                    typ = ev.get("type")
                    if typ == "session.output_audio.delta":
                        a = np.frombuffer(base64.b64decode(ev["delta"]), "<i2")
                        if first_audible is None and a.size and np.abs(a.astype(np.int32)).max() > 184:
                            first_audible = rel()
                        continue
                    log.append({"t": rel(), "type": typ, **({"error": ev["error"]} if typ == "error" else {})})
                    print(rel(), typ, ev.get("error", ""), ev.get("delta", ""))
                    if typ == "session.closed":
                        closed.set()
                        return
            except websockets.ConnectionClosed as e:
                log.append({"t": rel(), "type": "ws_closed", "detail": str(e)})
                print(rel(), "ws closed", e)
                closed.set()

        rd = asyncio.create_task(reader())
        await asyncio.sleep(0.5)
        await ws.send(json.dumps({"type": "session.update", "session": {"audio": {"output": {"language": "en"}}}}))
        print(f"idle for {idle:.0f} s with no audio ...")
        await asyncio.sleep(idle)
        if closed.is_set():
            print("session closed during idle -> keepalive IS needed")
        else:
            x, _ = load_wav(clip, 24000)
            pcm = to_pcm16(np.concatenate([x, np.zeros(24000 * 3, np.float32)]))
            step = 960
            t_start = rel()
            start = time.perf_counter()
            for i, k in enumerate(range(0, len(pcm), step)):
                await asyncio.sleep(max(0, start + i * 0.02 - time.perf_counter()))
                await ws.send(json.dumps({"type": "session.input_audio_buffer.append",
                                          "audio": base64.b64encode(pcm[k:k + step]).decode()}))
            await ws.send(json.dumps({"type": "session.close"}))
            await asyncio.wait_for(closed.wait(), 30)
            print(f"stream started at {t_start}s; first audible translation at {first_audible}s "
                  f"({None if first_audible is None else round(first_audible - t_start, 2)} s after stream start)")
        rd.cancel()
    out = ROOT / "results" / "probes" / f"openai_idle_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"idle_s": idle, "first_audible_s": first_audible, "events": log}, indent=1))
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", nargs="?", default=str(ROOT / "corpus" / "prepared" / "es_en_medium_1.wav"))
    ap.add_argument("--idle", type=float, default=90)
    a = ap.parse_args()
    asyncio.run(main(Path(a.clip), a.idle))
