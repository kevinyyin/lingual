"""OPTIONAL: Kyutai Hibiki (local, French -> English only). github.com/kyutai-labs/hibiki

Hibiki needs a CUDA GPU (PyTorch, `pip install moshi`, hf repo kyutai/hibiki-1b-pytorch-bf16)
or Apple Silicon (MLX, `pip install moshi_mlx`, kyutai/hibiki-1b-mlx-bf16). The reference
code (moshi.run_inference) is file-based; a streaming adapter must drive moshi's frame-level
generator at 12.5 Hz (80 ms frames) from send_audio() and emit each decoded output frame as
an audio_out event.

That streaming adapter has NOT been written: this harness was built on a machine with
neither CUDA nor Metal, so the moshi streaming API could not be exercised, and the
project rule is to never write provider code against an unverified API. On such machines
the provider is skipped cleanly and the reason appears in the report. To add it, implement
connect/send_audio/finish below on a supported machine, following moshi's own
run_inference.py, and add it to `bench run --providers`.
"""

from __future__ import annotations

import platform
import sys
from typing import Any

from bench.providers.base import Provider, ProviderUnavailable


def _hardware() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "metal"
    return None


class HibikiLocalProvider(Provider):
    name = "hibiki"
    input_sample_rate = 24000

    @classmethod
    def check_available(cls, cfg: dict[str, Any]) -> str | None:
        if not cfg.get("enabled", True):
            return "disabled in config"
        hw = _hardware()
        if hw is None:
            return "no CUDA or Apple Metal GPU on this machine (Hibiki needs one); skipped"
        return (f"{hw} available, but the streaming Hibiki adapter is not implemented yet "
                "(see bench/providers/hibiki_local.py); skipped")

    async def connect(self, source_lang: str, target_lang: str) -> None:
        raise ProviderUnavailable(self.check_available(self.cfg) or "hibiki unavailable")

    async def send_audio(self, pcm16_chunk: bytes, t_sent_ns: int) -> None:
        raise ProviderUnavailable("hibiki unavailable")

    async def finish(self) -> None:
        self._emit_closed()
