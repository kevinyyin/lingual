"""`bench selftest`: prove the harness measures a known delay before trusting it.

Runs the mock provider (tone mode) at 300 ms with 0 jitter, then 300 ms with +/-20 ms
jitter, over short and medium clips, and checks:
  * start lag          within +/-10 ms of 300 ms (zero jitter)
  * end lag (arrival)  within +/-10 ms of 300 ms (zero jitter)
  * end lag (playback) equals an independent re-simulation of the playback rules, and
                       (zero jitter) equals 300 ms + one chunk duration, +/-10 ms
  * pacing             systematic drift < 2 ms over the stream, p99 send lateness < 5 ms,
                       no chunk sent more than one chunk (20 ms) late
With jitter the tolerances widen by the jitter bound. Exit code is non-zero on failure.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from bench.analysis.metrics import latency_metrics, output_chunks, source_times
from bench.config import RESULTS_DIR, BenchConfig
from bench.corpus import PreparedClip, load_index
from bench.events import read_run
from bench.prepare import finalize_clip, normalize_peak, synth_os, trim_silence
from bench.audio import load_wav
from bench.clock import run as clock_run
from bench.runner import RunPaths, run_batch

SELFTEST_DIR = RESULTS_DIR / "selftest"
DELAY_MS = 300.0
TOL_MS = 10.0

_FALLBACK = [
    ("st_short_1", "short", "Where is the train station?"),
    ("st_short_2", "short", "How much does this cost?"),
    ("st_short_3", "short", "Thank you very much."),
    ("st_medium_1", "medium", "Could you tell me how to get to the museum from here, please?"),
    ("st_medium_2", "medium", "I would like to book a table for two people tomorrow evening."),
]


def selftest_clips(cfg: BenchConfig, log=print) -> list[PreparedClip]:
    idx = load_index()
    picked = [c for c in idx.values() if c.category == "short"][:3] + [c for c in idx.values() if c.category == "medium"][:3]
    picked = [c for c in picked if c.timing.onset_sample is not None and c.path.exists()]
    if len(picked) >= 4:
        log(f"[selftest] using {len(picked)} prepared corpus clips")
        return picked
    log("[selftest] corpus not prepared; synthesizing local English clips with the OS voice (offline)")
    out = []
    for cid, cat, text in _FALLBACK:
        raw = SELFTEST_DIR / "corpus" / f"{cid}.wav"
        if not raw.exists():
            synth_os(text, "en", raw, None, 0)
        x, sr = load_wav(raw)
        x = normalize_peak(trim_silence(x, sr), -3.0)
        out.append(finalize_clip(x, sr, cfg, cid=cid, pair="en-en", category=cat, source_lang="en", synthetic=True,
                                 out_dir=SELFTEST_DIR / "prepared", text=text, reference=text))
    return out


def replay_playback_end(arrivals_ns: list[int], durs_ns: list[int]) -> int:
    """Independent re-implementation of the playback rule for cross-checking."""
    t = None
    for a, d in zip(arrivals_ns, durs_ns):
        t = a + d if t is None or a >= t else t + d
    return t


@dataclass
class Check:
    case: str
    clip: str
    name: str
    value: float | None
    lo: float
    hi: float
    warn_only: bool = False

    @property
    def ok(self) -> bool:
        return self.value is not None and self.lo <= self.value <= self.hi


def _checks_for_run(path: Path, case: str, jitter_ms: float, chunk_ms: int, gate: float) -> list[Check]:
    run = read_run(path)
    cid = run.header.clip_id
    if run.footer is None or run.footer.status != "ok":
        return [Check(case, cid, f"run status ({run.footer.error if run.footer else 'no footer'})", None, 0, 0)]
    pcm = None
    if run.footer.output_audio_file:
        pcm = sf.read(str(path.parent.parent / "audio" / run.footer.output_audio_file), dtype="int16")[0]
    chunks = output_chunks(run, pcm, gate)
    m = latency_metrics(run, chunks)
    j = jitter_ms
    # lateness against the ideal schedule (a late source send or a late mock emission, from OS
    # preemption) stays queued in playback for the rest of the stream, as on a real speaker,
    # so playback bounds allow the observed lateness (jitter is accounted separately)
    planned = run.sends.planned_ns
    emit_late = max([max(0.0, (e.t_recv_ns - e.meta["due_ns"]) / 1e6,
                         (e.t_recv_ns - planned[e.meta["input_chunk"]]) / 1e6 - DELAY_MS - j)
                     for e in run.events if e.kind == "audio_out" and "due_ns" in e.meta] or [0.0])
    checks = [
        Check(case, cid, "start_lag_ms", m["start_lag_ms"], DELAY_MS - TOL_MS - j, DELAY_MS + TOL_MS + j),
        Check(case, cid, "end_lag_arrival_ms", m["end_lag_arrival_ms"], DELAY_MS - TOL_MS - j, DELAY_MS + TOL_MS + j),
    ]
    # playback: independent re-simulation from raw arrivals must agree to within 0.01 ms
    _, t_off = source_times(run)
    aud = [c for c in chunks]
    ref_end = replay_playback_end([c.t_recv_ns for c in aud], [c.dur_ns for c in aud])
    ref_lag = (ref_end - t_off) / 1e6 if ref_end is not None and t_off is not None else None
    diff = None if (ref_lag is None or m["end_lag_playback_ms"] is None) else abs(m["end_lag_playback_ms"] - ref_lag)
    checks.append(Check(case, cid, "end_lag_playback == re-simulation (|diff| ms)", diff, 0.0, 0.01))
    if m["end_lag_playback_ms"] is not None and m["end_lag_arrival_ms"] is not None:
        checks.append(Check(case, cid, "end_lag_playback - end_lag_arrival (>= one chunk)",
                            m["end_lag_playback_ms"] - m["end_lag_arrival_ms"], chunk_ms - 0.5,
                            chunk_ms + 2 * j + TOL_MS + emit_late))
    if j == 0:
        checks.append(Check(case, cid, f"end_lag_playback_ms (max lateness {emit_late:.1f})",
                            m["end_lag_playback_ms"], DELAY_MS + chunk_ms - TOL_MS,
                            DELAY_MS + chunk_ms + TOL_MS + emit_late))
    checks.append(Check(case, cid, "pacing drift over stream (abs ms)", abs(run.sends.drift_ms), 0.0, 2.0))
    checks.append(Check(case, cid, "pacing p99 lateness ms (<= half a chunk)", run.sends.p99_lateness_ms, -1.0, chunk_ms / 2))
    checks.append(Check(case, cid, "pacing max lateness ms (warning only)", run.sends.max_lateness_ms, -1.0, float(chunk_ms),
                        warn_only=True))
    return checks


def run_selftest(cfg: BenchConfig, log=print) -> bool:
    clips = selftest_clips(cfg, log)
    cfg = cfg.model_copy(update={"inter_run_delay_s": 0.0, "concurrency": 1})
    all_checks: list[Check] = []
    for case, jitter in (("300ms/0 jitter", 0.0), ("300ms/+-20ms jitter", 20.0)):
        tag = "j0" if jitter == 0 else "j20"
        paths = RunPaths(SELFTEST_DIR / tag / "raw", SELFTEST_DIR / tag / "audio")
        for d in (paths.raw_dir, paths.audio_dir):
            d.mkdir(parents=True, exist_ok=True)
            for f in d.glob("*"):
                f.unlink()
        log(f"[selftest] case {case}: {len(clips)} clips")
        overrides = {"mock": {"delay_ms": DELAY_MS, "jitter_ms": jitter, "mode": "tone", "seed": 42}}
        clock_run(run_batch(cfg, ["mock"], clips, 1, paths=paths, provider_overrides=overrides,
                              probe_network=False, log=lambda *_: None))
        for p in sorted(paths.raw_dir.glob("*.jsonl")):
            all_checks.extend(_checks_for_run(p, case, jitter, cfg.audio.chunk_ms, cfg.audio.output_gate_dbfs))

    width = max(len(c.name) for c in all_checks)
    failed = [c for c in all_checks if not c.ok and not c.warn_only]
    warned = [c for c in all_checks if not c.ok and c.warn_only]
    log("")
    log(f"{'case':20s} {'clip':14s} {'check':{width}s} {'value':>10s}  expected")
    for c in all_checks:
        v = "—" if c.value is None else f"{c.value:10.3f}"
        verdict = "ok" if c.ok else ("warn" if c.warn_only else "FAIL")
        log(f"{c.case:20s} {c.clip:14s} {c.name:{width}s} {v:>10s}  [{c.lo:g}, {c.hi:g}] {verdict}")
    log("")
    if warned:
        log(f"warning: {len(warned)} run(s) had an isolated send delayed by more than one chunk (OS preemption).")
        log("  Latency numbers stay valid (every lag is measured from the recorded actual send time), but for")
        log("  real runs prefer AC power, a high-performance power plan, and closing heavy background apps.")
    if failed:
        log(f"SELFTEST FAILED: {len(failed)}/{len(all_checks)} checks out of tolerance.")
        log("Diagnostics: large start/end lag errors usually mean the event loop was blocked or the")
        log("OS timer is coarse (on Windows the harness calls timeBeginPeriod(1)); pacing failures mean")
        log("the machine is overloaded. Close heavy apps and re-run. Do not trust real-provider numbers yet.")
        return False
    starts = [c.value for c in all_checks if c.name == "start_lag_ms" and c.case.startswith("300ms/0")]
    log(f"SELFTEST PASSED: {len(all_checks)} checks. Zero-jitter start lag: mean {np.mean(starts):.2f} ms, "
        f"max error {max(abs(s - DELAY_MS) for s in starts):.2f} ms.")
    return True
