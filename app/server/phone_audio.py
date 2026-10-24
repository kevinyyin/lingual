"""The phone's mic and speaker over the page's WebSocket, in place of the laptop's sounddevice ones.

ClientMic is MicTranscriber's `mic`; WsPlayer is what Voice's `player_factory` builds.
"""

import asyncio
import time
from typing import AsyncIterator, Awaitable, Callable

from . import stt, voice

SILENCE = bytes(stt.CHUNK * 2)  # 100 ms of PCM16
IDLE_S = 1.0  # keepalive only after a real gap: the phone sends a chunk every 128 ms, plus tunnel jitter
VOICE_BYTES_PER_S = voice.SAMPLE_RATE * 2


class ClientMic:
    """PCM16 16 kHz chunks pushed by the page; silence while none arrive so the Ink socket stays open.
    Only live audio: each new Ink connection starts empty, and at most MAX_CHUNKS wait while Ink is away."""

    MAX_CHUNKS = 8  # ~1 s of the phone's 128 ms chunks

    def __init__(self):
        self.q: asyncio.Queue[bytes] = asyncio.Queue()

    def push(self, b: bytes) -> None:
        if self.q.qsize() >= self.MAX_CHUNKS:
            self.q.get_nowait()  # drop the oldest
        self.q.put_nowait(b)

    async def __call__(self) -> AsyncIterator[bytes]:
        while not self.q.empty():  # queued while Ink was down: stale, maybe the plushie's own voice
            self.q.get_nowait()
        while True:
            try:
                yield await asyncio.wait_for(self.q.get(), IDLE_S)
            except TimeoutError:
                yield SILENCE


class WsPlayer:
    """Sends voice PCM to the page as it is generated, paced to real time: Voice's "done" (and the session leaving
    "speaking", which reopens the mic) must not come before the phone has nearly finished playing."""

    TAIL_S = 0.5  # after the phone finishes: its echo still travels the tunnel back and sits in the mic buffer

    def __init__(self, send: Callable[[bytes], Awaitable[None]], loop: asyncio.AbstractEventLoop, ahead_s: float = 0.3,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        self.send, self.loop, self.ahead_s, self.clock, self.sleep = send, loop, ahead_s, clock, sleep
        self.start: float | None = None
        self.sent_s = 0.0

    def write(self, b: bytes) -> None:
        """Called from a worker thread (Voice runs player.write in asyncio.to_thread)."""
        if self.start is None:
            self.start = self.clock()
        asyncio.run_coroutine_threadsafe(self.send(b), self.loop).result()
        self.sent_s += len(b) / VOICE_BYTES_PER_S
        lead = self.sent_s - (self.clock() - self.start)
        if lead > self.ahead_s:
            self.sleep(lead - self.ahead_s)

    def abort(self) -> None:
        """Cut off by a newer line: no wait (the phone plays out at most ahead_s of what was already sent)."""

    def close(self) -> None:
        """Blocks (Voice calls it in a worker thread) until the phone has played everything plus TAIL_S, so the
        session stays "speaking" (mic gated) while the plushie's own voice can still reach speech-to-text."""
        if self.start is not None:
            self.sleep(max(0.0, self.sent_s - (self.clock() - self.start)) + self.TAIL_S)
