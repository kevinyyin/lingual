"""Causal VAD on real speech with pauses of known length (no look-ahead)."""

from pathlib import Path

import numpy as np
import pytest

from bench.audio import load_wav, normalize_peak, room_tone, to_pcm16
from bench.prepare import trim_silence

AUDIO = Path(__file__).resolve().parent.parent / "corpus" / "audio"
SR = 24000


def speech(name):
    p = AUDIO / f"{name}.wav"
    if not p.exists():
        pytest.skip(f"corpus clip {name} not available")
    x, sr = load_wav(p, SR)
    return normalize_peak(trim_silence(x, SR), -3.0)


def stream(x, chunk_ms, min_silence_ms):
    from bench.causal_vad import CausalVAD

    vad = CausalVAD(SR, min_silence_ms)
    n = SR * chunk_ms // 1000
    ev = []
    for i in range(0, len(x), n):
        t = int(i / SR * 1e9)  # nominal send time of this chunk
        ev += vad.push(to_pcm16(x[i:i + n]), t)
    return ev


def build(pauses_s):
    """speech, pause, speech, pause, ... speech, 2 s tail. Returns signal and speech end times."""
    a, b = speech("es_en_short_1"), speech("es_en_short_2")
    parts, ends, t = [room_tone(SR // 2, SR, -60.0)], [], 0.5
    for i, p in enumerate(pauses_s + [2.0]):
        s = a if i % 2 == 0 else b
        parts.append(s)
        t += len(s) / SR
        ends.append(t)
        parts.append(room_tone(int(p * SR), SR, -60.0, seed=i + 1))
        t += p
    return np.concatenate(parts), ends


@pytest.mark.parametrize("min_silence_ms", [400, 600, 800])
def test_speech_end_fires_after_long_pauses_only(min_silence_ms):
    # pauses: 1.0 s (longer than every setting), 0.25 s (shorter than every setting)
    x, ends = build([1.0, 0.25])
    ev = stream(x, 20, min_silence_ms)
    se = [e for e in ev if e.kind == "speech_end"]
    # the 0.25 s pause must not split; the 1.0 s pause and the final 2.0 s tail must
    assert len(se) == 2, [(e.kind, e.t_detect_ns / 1e9) for e in ev]
    # detection happens after the true end + min_silence (causal), within ~150 ms of it
    for e, true_end in zip(se, [ends[0], ends[2]]):
        lag = e.t_detect_ns / 1e9 - true_end
        assert min_silence_ms / 1000 - 0.1 <= lag <= min_silence_ms / 1000 + 0.2, (min_silence_ms, lag)


def test_pause_between_settings_splits_only_short_settings():
    # a 0.5 s pause: splits at min_silence 400, not at 600 or 800
    x, _ = build([0.5])
    assert len([e for e in stream(x, 20, 400) if e.kind == "speech_end"]) == 2
    assert len([e for e in stream(x, 20, 800) if e.kind == "speech_end"]) == 1


def test_no_lookahead_events_never_precede_their_audio():
    x, _ = build([1.0])
    for chunk_ms in (20, 200):
        for e in stream(x, chunk_ms, 400):
            # event sample (16 kHz) must already have been sent at detection time
            assert e.sample_16k / 16000 <= e.t_detect_ns / 1e9 + chunk_ms / 1000 + 1e-6
