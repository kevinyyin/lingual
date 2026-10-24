"""Laptop mic -> Cartesia Ink 2 (turn-detecting STT) -> Session.on_transcript.

While the session isn't listening (signer turn, voice playing, avatar), silence is sent instead of the mic,
so the laptop's own voice output is never transcribed and the socket stays open.
"""

import asyncio
import logging
from typing import AsyncIterator, Awaitable, Callable

from .voice import WS_OPTIONS

log = logging.getLogger("asl.stt")
MODEL = "ink-2"
SAMPLE_RATE = 16000
CHUNK = 1600  # samples = 100 ms


async def default_mic() -> AsyncIterator[bytes]:
    import sounddevice as sd

    loop = asyncio.get_running_loop()
    q: asyncio.Queue[bytes] = asyncio.Queue()

    def callback(data, frames, time, status):
        loop.call_soon_threadsafe(q.put_nowait, bytes(data))

    with sd.RawInputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=CHUNK, callback=callback):
        while True:
            yield await q.get()


class MicTranscriber:
    def __init__(self, on_transcript: Callable[[str, bool], Awaitable[None]], is_open: Callable[[], bool],
                 client=None, api_key: str | None = None, mic: Callable[[], AsyncIterator[bytes]] = default_mic):
        if client is None:
            from cartesia import AsyncCartesia

            client = AsyncCartesia(api_key=api_key)
        self.client, self.on_transcript, self.is_open, self.mic = client, on_transcript, is_open, mic

    async def connection(self) -> None:
        """One Ink connection: stream audio and forward turn events until either side ends."""
        async with self.client.stt.auto_finalize.websocket(model=MODEL, encoding="pcm_s16le", sample_rate=SAMPLE_RATE,
                                                           websocket_connection_options=WS_OPTIONS) as ws:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._pump(ws))
                tg.create_task(self._listen(ws))

    async def _pump(self, ws) -> None:
        async for chunk in self.mic():
            await ws.send_raw(chunk if self.is_open() else bytes(len(chunk)))

    async def _listen(self, ws) -> None:
        async for ev in ws:
            if ev.type == "turn.update":
                await self.on_transcript(ev.transcript, False)
            elif ev.type == "turn.end":
                await self.on_transcript(ev.transcript, True)
            elif ev.type == "error":
                raise RuntimeError(getattr(ev, "message", "ink error"))

    async def run(self) -> None:
        """Reconnect forever; one failed connection must not end speech input."""
        while True:
            try:
                await self.connection()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                inner = e.exceptions if isinstance(e, ExceptionGroup) else [e]  # TaskGroup wraps the real cause
                log.warning("Ink connection ended: %s; reconnecting", "; ".join(f"{type(x).__name__}: {x}" for x in inner))
                await self.on_transcript("", True)  # end any half-heard turn so the session doesn't stick in listening
            await asyncio.sleep(1.0)
