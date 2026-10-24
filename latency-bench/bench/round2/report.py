"""Assemble results/round2/REPORT-ROUND2.md (IMRaD, shorter than round 1) from the
per-experiment sections, compute the decision gate, and append a JSON block of every
number used."""

from __future__ import annotations

import json
from pathlib import Path

from bench.round2.common import R2, SECTIONS, med

ORDER = ["exp1", "exp2ab", "exp2c", "exp3", "exp4", "exp5", "exp6", "prep78"]


def _load(name: str) -> dict:
    p = SECTIONS / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _costs() -> list[dict]:
    """Cost records, one per batch (a batch logged twice keeps its last record)."""
    p = R2 / "costs.jsonl"
    if not p.exists():
        return []
    by: dict[str, dict] = {}
    for x in p.read_text(encoding="utf-8").splitlines():
        if x.strip():
            c = json.loads(x)
            by[c["batch_id"]] = c
    return list(by.values())


def decision_gate() -> dict:
    gate: dict = {}
    e3 = _load("exp3").get("pooled", {})
    pause = {s: d for s, d in e3.items() if "pause" in s}
    if pause:
        hits = [s for s, d in pause.items() if d["median"] is not None and d["median"] <= 2300 and d["stalls"] == 0]
        gate["flush_on_pause_holds_2_3s_no_stalls"] = {"answer": bool(hits), "conditions_meeting": hits,
                                                       "per_condition": {s: {"median": d["median"], "stalls": d["stalls"],
                                                                             "max": d["max"]} for s, d in pause.items()}}
    else:
        gate["flush_on_pause_holds_2_3s_no_stalls"] = {"answer": None, "note": "experiment 3 not run"}
    cand = {}
    for name in ("exp5", "exp6"):
        d = _load(name)
        for key in ("5a", "5b", "summary"):
            for rec in d.get(key, []) or []:
                if rec.get("category") == "sentences" and str(rec.get("system", "")).startswith("cascade"):
                    cand[rec["system"]] = {"end_play_med": rec["end_play_med"], "chrf_med": rec["chrf_med"]}
    hits = [s for s, v in cand.items() if v["end_play_med"] is not None and v["end_play_med"] <= 2200
            and v["chrf_med"] is not None and v["chrf_med"] >= 85]
    gate["cascade_reaches_2_2s_at_chrf85"] = {"answer": bool(hits) if cand else None, "conditions_meeting": hits,
                                              "per_condition_sentences": cand,
                                              "note": "sentences = medium + noisy pooled, end lag at playback, 1x listener"}
    return gate


def assemble(intro_md: str, discussion_md: str, spend_actual_usd: float | None = None) -> Path:
    gate = decision_gate()
    costs = _costs()
    est = sum(c.get("openai_est_usd", 0) for c in costs)
    est_wall = sum(c.get("openai_est_usd_if_billed_by_wall_time", 0) for c in costs)
    parts = [intro_md.strip(), "", "# Results", ""]
    for n in ORDER:
        p = SECTIONS / f"{n}.md"
        if p.exists():
            parts += [p.read_text(encoding="utf-8").strip(), ""]
    g1 = gate["flush_on_pause_holds_2_3s_no_stalls"]
    g2 = gate["cascade_reaches_2_2s_at_chrf85"]
    parts += ["## Decision gate", "",
              f"1. **Does flush-on-pause hold the median at or under 2.3 s with no stalls?** "
              f"{'Not answered (experiment 3 not run).' if g1['answer'] is None else ('Yes: ' + ', '.join(g1['conditions_meeting']) if g1['answer'] else 'No.')}",
              f"2. **Does any cascade configuration reach 2.2 s at chrF 85 or better?** "
              f"{'Not answered.' if g2['answer'] is None else ('Yes: ' + ', '.join(g2['conditions_meeting']) if g2['answer'] else 'No (at a 1x listener; see experiment 1 for catch-up playback).')}",
              "", discussion_md.strip(), "",
              "# Spend", "",
              "| Batch | Experiment | OpenAI sessions | audio min | estimated $ (audio) | estimated $ if billed by session time |",
              "|---|---|---|---|---|---|"]
    for c in costs:
        parts.append(f"| {c['batch_id']} | {c['experiment']} | {c['openai_sessions']} | {c['openai_audio_min']:.1f} | "
                     f"{c['openai_est_usd']:.2f} | {c['openai_est_usd_if_billed_by_wall_time']:.2f} |")
    parts += ["", f"Estimated OpenAI translate spend: **${est:.2f}** (audio minutes × $0.034); **${est_wall:.2f}** if "
              "sessions were billed by open time. LLM, Deepgram and Cartesia usage are small and not itemized here.",
              f"Actual OpenAI spend from the usage dashboard: **{'$%.2f' % spend_actual_usd if spend_actual_usd is not None else 'TO FILL IN (paste the figure from the OpenAI usage dashboard)'}**.",
              "", "# Appendix · Data (JSON)", "", "```json"]
    data = {n: _load(n) for n in ORDER if (SECTIONS / f"{n}.json").exists()}
    data["decision_gate"] = gate
    data["costs"] = costs
    data["estimated_openai_usd"] = round(est, 2)
    parts += [json.dumps(data, indent=1, default=str), "```", ""]
    out = R2 / "REPORT-ROUND2.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    return out
