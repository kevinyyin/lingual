"""Mock round trip through the real runner: known delay in, same delay measured out."""

import numpy as np
import soundfile as sf

from bench import clock
from bench.analysis.metrics import latency_metrics, output_chunks
from bench.audio import room_tone, save_wav
from bench.config import BenchConfig
from bench.corpus import PreparedClip
from bench.events import SourceTiming, read_run
from bench.providers.mock import MockProvider
from bench.runner import Job, RunPaths, run_one

SR = 24000


def _clip(tmp_path) -> PreparedClip:
    """1 s room tone, 1.5 s loud 300 Hz burst, 2 s room tone. Ground-truth timing, no VAD."""
    x = room_tone(int(4.5 * SR), SR, -60.0)
    t = np.arange(int(1.5 * SR)) / SR
    x[SR: SR + len(t)] += 0.4 * np.sin(2 * np.pi * 300 * t)
    path = tmp_path / "burst.wav"
    save_wav(path, x, SR)
    return PreparedClip(id="burst", pair="es-en", category="short", source_lang="es", synthetic=True,
                        file=str(path), sample_rate=SR, duration_s=4.5, sha256="x",
                        timing=SourceTiming(sample_rate=SR, n_samples=len(x), onset_sample=SR,
                                            offset_sample=SR + len(t) - 1))


def _run(tmp_path, **mock_cfg):
    cfg = BenchConfig()
    paths = RunPaths(tmp_path / "raw", tmp_path / "audio")
    prov = MockProvider({"delay_ms": 300, "jitter_ms": 0, "mode": "echo", **mock_cfg})
    raw = clock.run(run_one(prov, Job("mock", _clip(tmp_path), 0), cfg, "test", paths, {}, {}))
    run = read_run(raw)
    pcm = sf.read(str(paths.audio_dir / run.footer.output_audio_file), dtype="int16")[0]
    return run, pcm, cfg


def test_mock_round_trip_measures_known_delay(tmp_path):
    run, pcm, cfg = _run(tmp_path)
    assert run.footer.status == "ok"
    m = latency_metrics(run, output_chunks(run, pcm, cfg.audio.output_gate_dbfs))
    assert abs(m["start_lag_ms"] - 300) <= 10, m
    assert abs(m["end_lag_arrival_ms"] - 300) <= 10, m
    # playback = 300 + one 20 ms chunk; any lateness against the ideal schedule (a late source
    # send or a late emission, from OS preemption) stays in the playback queue for the rest of
    # the stream, exactly as it would on a real speaker
    planned = run.sends.planned_ns
    late = max((e.t_recv_ns - planned[e.meta["input_chunk"]] - 300_000_000) / 1e6
               for e in run.events if e.kind == "audio_out")
    assert 310 <= m["end_lag_playback_ms"] <= 330 + max(0.0, late), (m, late)
    assert m["setup_ms"] is not None and m["setup_ms"] >= 50  # mock connect delay


def test_mock_echo_returns_the_input_audio(tmp_path):
    run, pcm, _ = _run(tmp_path)
    audio_events = [e for e in run.events if e.kind == "audio_out"]
    # one output chunk per input chunk inside the speech region (chunks 50..124)
    assert len(audio_events) == 75
    out = pcm.astype(np.float32) / 32768
    t = np.arange(len(out)) / SR
    ref = 0.4 * np.sin(2 * np.pi * 300 * t)
    corr = np.abs(np.dot(out[:SR], ref[:SR])) / (np.linalg.norm(out[:SR]) * np.linalg.norm(ref[:SR]))
    assert corr > 0.95  # echoed signal is the 300 Hz burst


def test_mock_jitter_preserves_order(tmp_path):
    run, _, _ = _run(tmp_path, jitter_ms=20, seed=3)
    t = [e.t_recv_ns for e in run.events if e.kind == "audio_out"]
    assert t == sorted(t)
    idx = [e.meta["input_chunk"] for e in run.events if e.kind == "audio_out"]
    assert idx == sorted(idx)
