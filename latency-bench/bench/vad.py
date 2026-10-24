"""Speech onset/offset on source clips with silero-vad."""

from __future__ import annotations

import numpy as np

from bench.audio import resample
from bench.config import VadCfg
from bench.events import SourceTiming

_VAD_SR = 16000
_model = None


def _get_model():
    global _model
    if _model is None:
        from silero_vad import load_silero_vad

        _model = load_silero_vad()
    return _model


def speech_segments(x: np.ndarray, sr: int, cfg: VadCfg) -> list[tuple[int, int]]:
    """Speech segments as (start, end) sample indices at the input rate `sr`."""
    import torch
    from silero_vad import get_speech_timestamps

    x16 = resample(x, sr, _VAD_SR)
    ts = get_speech_timestamps(
        torch.from_numpy(np.ascontiguousarray(x16)),
        _get_model(),
        sampling_rate=_VAD_SR,
        threshold=cfg.threshold,
        min_silence_duration_ms=cfg.min_silence_ms,
        speech_pad_ms=cfg.speech_pad_ms,
    )
    scale = sr / _VAD_SR
    return [(int(round(t["start"] * scale)), int(round(t["end"] * scale))) for t in ts]


def source_timing(x: np.ndarray, sr: int, cfg: VadCfg) -> SourceTiming:
    segs = speech_segments(x, sr, cfg)
    return SourceTiming(
        sample_rate=sr,
        n_samples=len(x),
        onset_sample=segs[0][0] if segs else None,
        offset_sample=segs[-1][1] if segs else None,
        segments=segs,
    )
