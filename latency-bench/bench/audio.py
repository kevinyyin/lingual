"""Audio I/O, resampling, padding, noise mixing, overlap mixing and chunking.

Internal representation: float32 mono numpy arrays in [-1, 1].
Wire representation: PCM16 little-endian mono bytes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import soxr


def load_wav(path: str | Path, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = data.mean(axis=1).astype(np.float32)
    if target_sr and sr != target_sr:
        x = resample(x, sr, target_sr)
        sr = target_sr
    return x, sr


def save_wav(path: str | Path, x: np.ndarray, sr: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(x, -1.0, 1.0), sr, subtype="PCM_16")


def save_pcm16_wav(path: str | Path, pcm: bytes, sr: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.frombuffer(pcm, dtype="<i2"), sr, subtype="PCM_16")


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    return soxr.resample(x, sr_in, sr_out, quality="HQ").astype(np.float32)


def to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767.0).round().astype("<i2").tobytes()


def from_pcm16(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype="<i2").astype(np.float32) / 32768.0


def dbfs_to_amp(dbfs: float) -> float:
    return float(10 ** (dbfs / 20.0))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0


def room_tone(n: int, sr: int, dbfs: float, seed: int = 0) -> np.ndarray:
    """Low-level pink-ish noise (never digital zero), RMS normalized to `dbfs`."""
    if n <= 0:
        return np.zeros(0, np.float32)
    rng = np.random.default_rng(seed)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / sr)
    shape = 1.0 / np.sqrt(np.maximum(f, 20.0))  # 1/f power above 20 Hz
    shape[f > 8000] *= 0.3                       # roll off the top like a real room
    out = np.fft.irfft(spec * shape, n)
    out = out / (rms(out) + 1e-12) * dbfs_to_amp(dbfs)
    return out.astype(np.float32)


_TONE_CACHE: dict[tuple[int, int, float, int], np.ndarray] = {}


def room_tone_cached(n: int, sr: int, dbfs: float, seed: int = 0) -> np.ndarray:
    key = (n, sr, dbfs, seed)
    if key not in _TONE_CACHE:
        _TONE_CACHE[key] = room_tone(n, sr, dbfs, seed)
    return _TONE_CACHE[key]


def pad_with_room_tone(x: np.ndarray, sr: int, pre_s: float, post_s: float, dbfs: float, seed: int = 0) -> np.ndarray:
    n_pre, n_post = int(round(pre_s * sr)), int(round(post_s * sr))
    bed = room_tone(n_pre + len(x) + n_post, sr, dbfs, seed)
    y = bed.copy()
    y[n_pre : n_pre + len(x)] += x
    return y


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    """Mix noise under speech at the given SNR (speech RMS over the whole clip)."""
    rng = np.random.default_rng(seed)
    if len(noise) < len(speech):
        reps = int(np.ceil(len(speech) / len(noise)))
        noise = np.tile(noise, reps)
    start = int(rng.integers(0, len(noise) - len(speech) + 1))
    seg = noise[start : start + len(speech)]
    gain = rms(speech) / (rms(seg) + 1e-12) / (10 ** (snr_db / 20.0))
    y = speech + seg * gain
    peak = np.max(np.abs(y))
    return (y / peak * 0.95).astype(np.float32) if peak > 0.99 else y.astype(np.float32)


def mix_overlap(a: np.ndarray, b: np.ndarray, sr: int, overlap_s: float) -> tuple[np.ndarray, int]:
    """Speaker B starts `overlap_s` before speaker A finishes. Returns (mix, b_start_sample)."""
    ov = min(int(round(overlap_s * sr)), len(a), len(b))
    b_start = len(a) - ov
    y = np.zeros(max(len(a), b_start + len(b)), dtype=np.float32)
    y[: len(a)] += a
    y[b_start : b_start + len(b)] += b
    peak = np.max(np.abs(y))
    if peak > 0.99:
        y = y / peak * 0.95
    return y, b_start


def normalize_peak(x: np.ndarray, peak_dbfs: float = -3.0) -> np.ndarray:
    p = float(np.max(np.abs(x))) if len(x) else 0.0
    return x if p == 0 else (x / p * dbfs_to_amp(peak_dbfs)).astype(np.float32)


def chunk_pcm16(x: np.ndarray, sr: int, chunk_ms: int, tail_fill: np.ndarray | None = None) -> list[bytes]:
    """Split into fixed-size PCM16 chunks; the final partial chunk is completed with
    `tail_fill` samples (room tone) rather than digital zero."""
    n = sr * chunk_ms // 1000
    total = int(np.ceil(len(x) / n)) * n
    if total > len(x):
        fill = tail_fill[: total - len(x)] if tail_fill is not None and len(tail_fill) >= total - len(x) else np.zeros(total - len(x), np.float32)
        x = np.concatenate([x, fill.astype(np.float32)])
    pcm = to_pcm16(x)
    step = n * 2
    return [pcm[i : i + step] for i in range(0, len(pcm), step)]


def peak_dbfs_pcm16(b: bytes) -> float:
    a = np.frombuffer(b, dtype="<i2")
    if len(a) == 0:
        return -np.inf
    p = int(np.max(np.abs(a.astype(np.int32))))
    return 20 * np.log10(p / 32768.0) if p > 0 else -np.inf
