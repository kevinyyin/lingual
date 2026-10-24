"""Abstract provider interface. The runner treats every provider identically."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from bench.clock import now_ns
from bench.events import ProviderEvent


class UnsupportedPair(Exception):
    """Raised by connect() when a provider cannot serve a language pair (recorded, not a failure)."""


class ProviderUnavailable(Exception):
    """Raised when a provider cannot run on this machine / without credentials."""


@dataclass
class ClipContext:
    """Ground truth about the clip being streamed. Real providers must ignore this;
    only the mock uses it (to 'speak' exactly during the VAD speech region)."""

    onset_chunk: int | None
    offset_chunk: int | None
    chunk_ms: int


class Provider(ABC):
    name: str = "base"
    input_sample_rate: int = 16000
    hosts: tuple[str, ...] = ()  # hostnames probed for TCP/TLS timing before a batch

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self._queue: asyncio.Queue[ProviderEvent] = asyncio.Queue()
        self._closed_emitted = False

    # ---- optional hooks -------------------------------------------------
    @classmethod
    def check_available(cls, cfg: dict[str, Any]) -> str | None:
        """Return None if runnable, else a human-readable reason it is skipped."""
        return None

    def set_clip_context(self, ctx: ClipContext) -> None:  # noqa: B027 - intentional no-op
        pass

    def public_config(self) -> dict[str, Any]:
        """Config echoed into the run header (no secrets)."""
        return {k: v for k, v in self.cfg.items() if "key" not in k.lower() and not k.startswith("_")}

    # ---- required interface --------------------------------------------
    @abstractmethod
    async def connect(self, source_lang: str, target_lang: str) -> None:
        """Open the session; return once it is ready to accept audio. Must emit a
        `session_ready` event stamped when the readiness signal arrived."""

    @abstractmethod
    async def send_audio(self, pcm16_chunk: bytes, t_sent_ns: int) -> None: ...

    @abstractmethod
    async def finish(self) -> None:
        """Source ended: flush/close gracefully and wait until all output has drained.
        Must result in a `closed` event."""

    async def aclose(self) -> None:
        """Hard close after an error. Must be idempotent."""
        self._emit_closed()

    async def events(self) -> AsyncIterator[ProviderEvent]:
        while True:
            ev = await self._queue.get()
            yield ev
            if ev.kind == "closed":
                return

    # ---- helpers ----------------------------------------------------------
    def _emit(self, ev: ProviderEvent) -> None:
        if ev.kind == "closed":
            if self._closed_emitted:
                return
            self._closed_emitted = True
        self._queue.put_nowait(ev)

    def _emit_closed(self, **meta: Any) -> None:
        self._emit(ProviderEvent("closed", now_ns(), meta=meta))

    def _mark(self, stage: str, t_ns: int | None = None, **meta: Any) -> None:
        self._emit(ProviderEvent("stage_mark", t_ns if t_ns is not None else now_ns(), stage=stage, meta=meta))
