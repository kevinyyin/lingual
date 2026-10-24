"""Live check of the Cartesia shapes voice.py and stt.py rely on. Saves nothing.

Run: cd app && uv run python scripts/probe_cartesia.py   (needs CARTESIA_API_KEY in app/.env)
"""

import asyncio
import os
import time
from pathlib import Path

from cartesia import AsyncCartesia
from dotenv import load_dotenv

from server.voice import WS_OPTIONS

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
TTS_MODEL, STT_MODEL, VOICE = "sonic-3.6", "ink-2", "0834f3df-e650-4766-a20c-5a93a43aa6e3"  # Leo: emotive voice


async def main() -> None:
    client = AsyncCartesia(api_key=os.environ["CARTESIA_API_KEY"])

    t0 = time.perf_counter()
    first, kinds, pcm = None, set(), bytearray()
    async with client.tts.websocket_connect(websocket_connection_options=WS_OPTIONS) as conn:
        ctx = conn.context(model_id=TTS_MODEL, voice=VOICE,
                           output_format={"container": "raw", "encoding": "pcm_s16le", "sample_rate": 16000},
                           generation_config={"emotion": "sad", "speed": 0.85, "volume": 0.9})
        await ctx.push("I'm sorry, we are out of oat milk today.")
        await ctx.no_more_inputs()
        async for ev in ctx.receive():
            kinds.add(ev.type)
            if ev.type == "chunk" and ev.audio:
                first = first or time.perf_counter() - t0
                pcm += ev.audio
    print(f"TTS events {sorted(kinds)}; first audio {first * 1000:.0f} ms; {len(pcm)} bytes")

    t0 = time.perf_counter()
    async with client.stt.auto_finalize.websocket(model=STT_MODEL, encoding="pcm_s16le", sample_rate=16000,
                                                  keyterm=["oat milk"], websocket_connection_options=WS_OPTIONS) as ws:
        for i in range(0, len(pcm), 3200):  # 100 ms chunks
            await ws.send_raw(bytes(pcm[i:i + 3200]))
            await asyncio.sleep(0.1)
        await ws.send({"type": "close"})
        async for ev in ws:
            print(f"STT {ev.type} @{(time.perf_counter() - t0) * 1000:.0f} ms {getattr(ev, 'transcript', '')!r}")
            if ev.type in ("turn.end", "error"):
                break


if __name__ == "__main__":
    asyncio.run(main())
