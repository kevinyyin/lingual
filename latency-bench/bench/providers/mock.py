"""Fixed-delay mock provider, used to prove the harness measures latency correctly.

For every input chunk inside the source speech region (given by ClipContext), the mock
emits one output chunk of the same duration exactly `delay_ms` (+ uniform jitter in
[-jitter_ms, +jitter_ms]) after that chunk was sent. Output order is preserved: a chunk
is never delivered before the previous one.

Modes:
  echo  output = the input chunk itself (resampled to output_sample_rate)
  tone  output = a 440 Hz tone at -12 dBFS; makes the output onset unambiguous so the
        selftest checks timing only, independent of the output silence gate.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

import numpy as np

from bench.audio import from_pcm16, resample, to_pcm16
from bench.clock import now_ns, sleep_until_ns
from bench.events import ProviderEvent
from bench.providers.base import ClipContext, Provider


class MockProvider(Provider):
    name = "mock"

    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)
        self.delay_ns = int(float(cfg.get("delay_ms", 300)) * 1e6)
        self.jitter_ns = int(float(cfg.get("jitter_ms", 0)) * 1e6)
        self.mode = cfg.get("mode", "echo")
        self.input_sample_rate = int(cfg.get("input_sample_rate", 16000))
        self.output_sample_rate = int(cfg.get("output_sample_rate", 24000))
        self.connect_delay_ns = int(float(cfg.get("connect_delay_ms", 50)) * 1e6)
        self._rng = random.Random(cfg.get("seed", 0))
        self._ctx: ClipContext | None = None
        self._pending: asyncio.Queue[tuple[int, int, bytes] | None] = asyncio.Queue()
        self._emitter: asyncio.Task | None = None
        self._chunk_index = 0
        self._last_due = 0

    def set_clip_context(self, ctx: ClipContext) -> None:
        self._ctx = ctx

    async def connect(self, source_lang: str, target_lang: str) -> None:
        await sleep_until_ns(now_ns() + self.connect_delay_ns)
        self._emit(ProviderEvent("session_ready", now_ns()))
        self._emitter = asyncio.create_task(self._emit_loop())

    def _in_region(self, i: int) -> bool:
        c = self._ctx
        if c is None:
            return True
        if c.onset_chunk is None or c.offset_chunk is None:
            return False
        return c.onset_chunk <= i <= c.offset_chunk

    async def send_audio(self, pcm16_chunk: bytes, t_sent_ns: int) -> None:
        i = self._chunk_index
        self._chunk_index += 1
        if self._in_region(i):
            self._pending.put_nowait((i, t_sent_ns, pcm16_chunk))

    def _render(self, chunk: bytes) -> bytes:
        n_out = int(round(len(chunk) // 2 * self.output_sample_rate / self.input_sample_rate))
        if self.mode == "tone":
            t = np.arange(n_out) / self.output_sample_rate
            return to_pcm16(0.25 * np.sin(2 * np.pi * 440.0 * t))
        x = resample(from_pcm16(chunk), self.input_sample_rate, self.output_sample_rate)
        return to_pcm16(x[:n_out])

    async def _emit_loop(self) -> None:
        while True:
            item = await self._pending.get()
            if item is None:
                break
            i, t_sent, chunk = item
            jitter = self._rng.randint(-self.jitter_ns, self.jitter_ns) if self.jitter_ns else 0
            due = max(self._last_due, t_sent + self.delay_ns + jitter)
            self._last_due = due
            out = self._render(chunk)
            await sleep_until_ns(due)
            self._emit(
                ProviderEvent(
                    "audio_out", now_ns(), payload_size=len(out), pcm=out,
                    sample_rate=self.output_sample_rate, meta={"input_chunk": i, "due_ns": due},
                )
            )

    async def finish(self) -> None:
        self._pending.put_nowait(None)
        if self._emitter:
            await self._emitter
        self._emit_closed()

    async def aclose(self) -> None:
        if self._emitter and not self._emitter.done():
            self._emitter.cancel()
        self._emit_closed()
