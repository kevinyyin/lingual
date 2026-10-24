"""Experiment 1: catch-up playback, re-scored offline on round-1 runs (no API calls).

Every round-1 run is replayed through the playback simulator under each policy; only the
simulated listener changes, never the recorded arrivals. A listening check renders the
actual time-stretched output (pitch-preserving WSOLA via audiotsm) for 3 runs per system,
transcribes it with faster-whisper and scores chrF against the unstretched output's
transcript to flag intelligibility loss.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from bench.analysis.metrics import (
    PlaybackPolicy, output_chunks, policy_end_lag_ms, simulate_playback_policy, source_times, utterance_times,
)
from bench.audio import save_wav
from bench.config import BenchConfig
from bench.round2.common import (
    R2, ROUND1_BATCHES, boot_ci_median, fmt_ci, fmt_ms, load_pcm, med, raw_paths, system_label, write_section,
)
from bench.events import read_run

POLICIES = [PlaybackPolicy("none")] + [
    PlaybackPolicy("catchup", thr, sp) for sp in (1.15, 1.25, 1.4) for thr in (300, 500)
] + [PlaybackPolicy("catchup_end", 500, 1.25)]
CATS = ["short", "medium", "noisy", "long", "overlap", "same_language"]
OUT = R2 / "exp1"


def tail_windows(run) -> list[tuple[int, int]]:
    utts = utterance_times(run)
    if not utts:
        _, off = source_times(run)
        return [(off, 2**62)] if off is not None else []
    return [(utts[i][1], utts[i + 1][0] if i + 1 < len(utts) else 2**62) for i in range(len(utts))]


def rescore(cfg: BenchConfig, log=print) -> list[dict]:
    rows = []
    for p in raw_paths(ROUND1_BATCHES):
        run = read_run(p)
        if not run.ok:
            continue
        pcm = load_pcm(run)
        chunks = output_chunks(run, pcm, cfg.audio.output_gate_dbfs)
        _, t_off = source_times(run)
        win = tail_windows(run)
        row = {"run_id": run.header.run_id, "system": system_label(run), "clip_id": run.header.clip_id,
               "category": run.header.category, "rep": run.header.repetition}
        for pol in POLICIES:
            blocks = simulate_playback_policy(chunks, pol, win)
            row[pol.label] = policy_end_lag_ms(chunks, blocks, t_off)
            row[pol.label + "|fast_s"] = round(sum((b.n / b.sr) for b in blocks if b.speed > 1), 3)
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "rescored.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    log(f"[exp1] re-scored {len(rows)} round-1 runs under {len(POLICIES)} policies")
    return rows


def render_policy_audio(pcm: np.ndarray, chunks, blocks, sr: int) -> np.ndarray:
    """Render what the simulated listener hears: groups of contiguous same-speed blocks,
    stretched with WSOLA (pitch preserved), placed at their play times; silence between."""
    from audiotsm import wsola
    from audiotsm.io.array import ArrayReader, ArrayWriter

    if not blocks:
        return np.zeros(0, np.float32)
    t0 = blocks[0].start_ns
    total = int((blocks[-1].end_ns - t0) * sr / 1e9) + sr
    out = np.zeros(total, np.float32)
    x = pcm.astype(np.float32) / 32768.0
    groups, cur = [], [blocks[0]]
    for b in blocks[1:]:
        p = cur[-1]
        if b.speed == p.speed and abs(b.start_ns - p.end_ns) < 1e6 and b.offset == p.offset + p.n:
            cur.append(b)
        else:
            groups.append(cur)
            cur = [b]
    groups.append(cur)
    for g in groups:
        seg = x[g[0].offset: g[-1].offset + g[-1].n]
        sp = g[0].speed
        if sp != 1.0 and len(seg) > 2048:
            padded = np.concatenate([seg, np.zeros(4096, np.float32)])[None, :]  # avoid WSOLA tail loss
            w = ArrayWriter(1)
            wsola(1, speed=sp).run(ArrayReader(padded), w)
            seg = w.data[0][: int(round(len(seg) / sp))]
        start = int((g[0].start_ns - t0) * sr / 1e9)
        out[start: start + len(seg)] += seg[: max(0, total - start)]
    return out


def listening_check(cfg: BenchConfig, rows: list[dict], log=print) -> list[dict]:
    from bench.analysis.align import transcribe_words
    from bench.analysis.quality import chrf

    out_dir = OUT.parent / "catchup_audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    picks = []
    for system in sorted({r["system"] for r in rows}):
        cand = [r for r in rows if r["system"] == system and r["category"] == "medium" and r["rep"] == 0]
        picks += sorted(cand, key=lambda r: r["clip_id"])[:3]
    checks = []
    by_id = {p.stem: p for p in raw_paths(ROUND1_BATCHES)}
    for r in picks:
        run = read_run(by_id[r["run_id"]])
        pcm = load_pcm(run)
        sr = run.footer.output_sample_rate
        chunks = output_chunks(run, pcm, cfg.audio.output_gate_dbfs)
        ref = transcribe_words(cfg, pcm.astype(np.float32) / 32768.0, sr, "en")["text"]
        for pol in (PlaybackPolicy("catchup", 500, 1.25), PlaybackPolicy("catchup", 300, 1.4)):
            blocks = simulate_playback_policy(chunks, pol, tail_windows(run))
            y = render_policy_audio(pcm, chunks, blocks, sr)
            name = f"{r['run_id']}__{pol.label.replace('@', '_').replace('/', '_')}.wav"
            save_wav(out_dir / name, y, sr)
            hyp = transcribe_words(cfg, y, sr, "en")["text"]
            fast_s = round(sum(b.n / b.sr for b in blocks if b.speed > 1), 2)
            checks.append({"system": r["system"], "run_id": r["run_id"], "policy": pol.label, "file": name,
                           "sped_up_content_s": fast_s, "chrf_vs_unstretched": chrf(hyp, ref),
                           "unstretched_asr": ref, "stretched_asr": hyp})
            log(f"[exp1] {r['system']:22s} {pol.label:22s} sped {fast_s:5.2f}s chrF {checks[-1]['chrf_vs_unstretched']}")
    (OUT / "listening_check.json").write_text(json.dumps(checks, indent=1, ensure_ascii=False), encoding="utf-8")
    return checks


def summarize(rows: list[dict]) -> dict:
    res: dict = {}
    for system in sorted({r["system"] for r in rows}):
        for cat in CATS:
            rs = [r for r in rows if r["system"] == system and r["category"] == cat and r["none"] is not None]
            if not rs:
                continue
            clips = len({r["clip_id"] for r in rs})
            d = {"n_runs": len(rs), "n_clips": clips}
            for pol in POLICIES:
                vals = [r[pol.label] for r in rs]
                deltas = [r[pol.label] - r["none"] for r in rs]
                d[pol.label] = {"median": med(vals), "ci95": boot_ci_median(vals),
                                "median_delta": med(deltas), "delta_ci95": boot_ci_median(deltas)}
            res[f"{system}|{cat}"] = d
    return res


def write_report(summary: dict, checks: list[dict]) -> None:
    pol_cols = [p.label for p in POLICIES]
    lines = ["## Experiment 1 · Catch-up playback (offline re-score of round 1)", "",
             "Every successful round-1 run (209) was replayed through the playback simulator under each policy. "
             "Recorded arrivals are unchanged; only the simulated listener speeds up when queued audio exceeds the "
             "threshold (and returns to 1x below half of it). `catchup_end` allows speed-up only after a source "
             "utterance offset. Values: median end lag at playback (ms); in parentheses the median per-run change "
             "vs `none`. n = runs / independent clips.", "",
             "| System · category | n | " + " | ".join(pol_cols) + " |",
             "|---|---|" + "---|" * len(pol_cols)]
    for key, d in summary.items():
        system, cat = key.split("|")
        cells = []
        for p in pol_cols:
            x = d[p]
            cells.append(fmt_ms(x["median"]) if p == "none" else f"{fmt_ms(x['median'])} ({x['median_delta']:+,.0f})")
        lines.append(f"| {system} · {cat} | {d['n_runs']}/{d['n_clips']} | " + " | ".join(cells) + " |")
    lines += ["", "95% bootstrap CIs of the per-run change for `catchup@1.25x/500ms` (runs resampled; "
              "clips are few, so these understate uncertainty):", ""]
    for key, d in summary.items():
        x = d["catchup@1.25x/500ms"]
        lines.append(f"- {key.replace('|', ' · ')}: {x['median_delta']:+,.0f} ms {fmt_ci(x['delta_ci95'])} "
                     f"(n = {d['n_runs']} runs, {d['n_clips']} clips)")
    lines += ["", "**Listening check** (pitch-preserving WSOLA, `audiotsm` 0.1.2): chrF of the Whisper transcript of "
              "the stretched rendering against the transcript of the unstretched output. 100 = identical words.", "",
              "| System | Run | Policy | Content sped up (s) | chrF vs unstretched |", "|---|---|---|---|---|"]
    for c in checks:
        lines.append(f"| {c['system']} | {c['run_id'].split('_', 2)[2]} | {c['policy']} | {c['sped_up_content_s']} | "
                     f"{c['chrf_vs_unstretched']} |")
    lines += ["", "Rendered files: `results/round2/catchup_audio/`."]
    write_section("exp1", "\n".join(lines), {"summary": summary, "listening_check": checks})


def run(cfg: BenchConfig, log=print) -> None:
    rows = rescore(cfg, log)
    summary = summarize(rows)
    checks = listening_check(cfg, rows, log)
    write_report(summary, checks)
    log("[exp1] wrote results/round2/sections/exp1.md")
