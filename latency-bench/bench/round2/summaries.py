"""Per-condition summary tables shared by round-2 experiment sections."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from bench.round2.common import boot_ci_median, fmt_ci, fmt_ms, med, p90, paired_clip_diffs, wilson

ROUND1_LABEL = {"openai": "openai", "cascade_incremental": "cascade[incremental]"}


def primary_end_lag(r: dict) -> float | None:
    """Pooled per-utterance median for multi-utterance clips, else the run's end lag."""
    return r.get("utt_end_lag_playback_ms") if r.get("utterances") is not None and r["category"] in (
        "two_utterances", "hesitation", "three_turns") else r.get("end_lag_playback_ms")


def condition_summary(rows: list[dict], categories: list[str] | None = None) -> list[dict[str, Any]]:
    """One record per (system, category) plus a pooled 'sentences' row (medium + noisy)."""
    out = []
    by: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by[(r["system"], r["category"])].append(r)
        if r["category"] in ("medium", "noisy"):
            by[(r["system"], "sentences")].append(r)
    for (system, cat), rs in sorted(by.items()):
        if categories and cat not in categories:
            continue
        ok = [r for r in rs if r["status"] == "ok"]
        att = [r for r in rs if r["status"] in ("ok", "failed")]
        stalls = sum(1 for r in ok if r.get("stall"))
        held = sum(1 for r in ok if r.get("held_until_flush"))
        end = [primary_end_lag(r) for r in ok]
        rec = {
            "system": system, "category": cat, "n_runs": len(ok), "n_clips": len({r["clip_id"] for r in ok}),
            "n_failed": len(att) - len(ok),
            "start_lag_med": med(r.get("start_lag_ms") for r in ok), "start_lag_ci": boot_ci_median([r.get("start_lag_ms") for r in ok]),
            "end_arrival_med": med(r.get("end_lag_arrival_ms") for r in ok),
            "end_play_med": med(end), "end_play_ci": boot_ci_median(end), "end_play_p90": p90(end),
            "end_play_max": max([x for x in end if x is not None], default=None),
            "stalls": stalls, "stall_rate": stalls / len(ok) if ok else None,
            "stall_ci": wilson(stalls, len(ok)) if ok else None, "held_until_flush": held,
            "ratio_span_med": med(r.get("ratio_span") for r in ok), "chrf_med": med(r.get("chrf") for r in ok),
        }
        out.append(rec)
    return out


def markdown_table(recs: list[dict], title_cols: str = "Condition · category") -> str:
    lines = [f"| {title_cols} | runs/clips | start lag med | end lag arrival med | end lag playback med [95% CI] | "
             "p90 | max | stalls (rate, Wilson 95%) | output/source | chrF |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        sc = r["stall_ci"]
        st = (f"{r['stalls']}/{r['n_runs']} ({100 * r['stall_rate']:.0f}%, {100 * sc[0]:.0f}–{100 * sc[1]:.0f}%)"
              if r["stall_rate"] is not None else "—")
        ratio = "—" if r["ratio_span_med"] is None else f"{r['ratio_span_med']:.2f}"
        chrf_s = "—" if r["chrf_med"] is None else f"{r['chrf_med']:.0f}"
        lines.append(
            f"| {r['system']} · {r['category']} | {r['n_runs']}/{r['n_clips']} | {fmt_ms(r['start_lag_med'])} | "
            f"{fmt_ms(r['end_arrival_med'])} | {fmt_ms(r['end_play_med'])} {fmt_ci(r['end_play_ci'])} | "
            f"{fmt_ms(r['end_play_p90'])} | {fmt_ms(r['end_play_max'])} | {st} | {ratio} | {chrf_s} |")
    return "\n".join(lines)


def paired_vs_round1(rows: list[dict], round1: list[dict], round1_label: str, metric: str = "end_lag_playback_ms",
                     categories: tuple[str, ...] = ("short", "medium", "noisy", "long")) -> dict[str, dict]:
    """Per-clip (median condition - median round-1) for each system, same clip ids."""
    base = [r for r in round1 if r.get("provider_label") == round1_label and r["status"] == "ok"
            and r["category"] in categories]
    res = {}
    for system in sorted({r["system"] for r in rows}):
        rs = [r for r in rows if r["system"] == system and r["status"] == "ok" and r["category"] in categories]
        d = paired_clip_diffs(rs, base, metric)
        if d:
            res[system] = {"per_clip_ms": d, "median_ms": float(np.median(list(d.values()))), "n_clips": len(d),
                           "n_faster": sum(1 for v in d.values() if v < 0)}
    return res
