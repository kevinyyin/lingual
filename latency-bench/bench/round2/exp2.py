"""Experiment 2: OpenAI stall study (sections 2a, 2b, 2c)."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from bench.round2.analyze import load_rows
from bench.round2.common import round1_rows, wilson, write_section
from bench.round2.summaries import condition_summary, markdown_table, paired_vs_round1

STALL_DEF = ("A **stall** is a run whose start lag exceeds 5 s, or that produced no audible output before the harness "
             "began closing the session (`session.close`, sent 30 s after speech ended). **Held until flush** = audible "
             "output exists, but only after `session.close`.")


def stall_table(rows: list[dict]) -> tuple[str, dict]:
    by = defaultdict(list)
    for r in rows:
        by[r["system"]].append(r)
        by[f"{r['system']} · {r['category']}"].append(r)
    lines = ["| Condition | runs ok | failed | stalls | stall rate [Wilson 95%] | held until flush | "
             "start lag p10 / p50 / p90 / max (ms) |", "|---|---|---|---|---|---|---|"]
    data = {}
    for k in sorted(by):
        rs = by[k]
        ok = [r for r in rs if r["status"] == "ok"]
        k_st = sum(1 for r in ok if r.get("stall"))
        held = sum(1 for r in ok if r.get("held_until_flush"))
        lo, hi = wilson(k_st, len(ok))
        sl = [r["start_lag_ms"] for r in ok if r.get("start_lag_ms") is not None]
        q = [float(np.percentile(sl, p)) for p in (10, 50, 90)] + [max(sl)] if sl else []
        data[k] = {"n": len(ok), "failed": len(rs) - len(ok), "stalls": k_st, "rate": k_st / len(ok) if ok else None,
                   "wilson": (lo, hi), "held": held, "start_lag_q": q}
        lines.append(f"| {k} | {len(ok)} | {len(rs) - len(ok)} | {k_st} | "
                     f"{100 * k_st / max(1, len(ok)):.1f}% [{100 * lo:.1f}–{100 * hi:.1f}%] | {held} | "
                     + (" / ".join(f"{v:,.0f}" for v in q) if q else "—") + " |")
    return "\n".join(lines), data


def stalled_list(rows: list[dict]) -> str:
    st = [r for r in rows if r.get("stall")]
    if not st:
        return "No stalled runs."
    lines = ["| Condition | Clip | rep | location | start lag (ms) | held until flush | close sent (s after offset) | "
             "end lag arrival (ms) | kind | output transcript |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(st, key=lambda r: (r["system"], r["clip_id"], r["repetition"])):
        kind = ("text without audible speech" if not r.get("produced_output") and r.get("provider_output_text")
                else "held until flush" if r.get("held_until_flush") else "late start")
        sl = r.get("start_lag_ms")
        el = r.get("end_lag_arrival_ms")
        lines.append(f"| {r['system']} | {r['clip_id']} | {r['repetition']} | {(r.get('location') or '')[:12]} | "
                     f"{'—' if sl is None else f'{sl:,.0f}'} | {r.get('held_until_flush')} | "
                     f"{(r.get('t_close_rel_offset_ms') or 0) / 1000:.1f} | {'—' if el is None else f'{el:,.0f}'} | "
                     f"{kind} | {(r.get('provider_output_text') or '')[:70]} |")
    return "\n".join(lines)


def best_condition(data: dict, systems: list[str]) -> str:
    """Lowest stall rate; ties broken by lower median start lag."""
    cand = [(data[s]["rate"], data[s]["start_lag_q"][1] if data[s]["start_lag_q"] else 1e9, s) for s in systems]
    return sorted(cand)[0][2]


def write_2ab() -> dict:
    a = load_rows("exp2a")
    b = load_rows("exp2b")
    parts = ["## Experiment 2 · OpenAI stall study", "", STALL_DEF, "",
             "Both phases stream short and medium clips (6 clips) with a 30 s trailing pad. They run at concurrency 4 "
             "because they count events; their milliseconds are not used as latency medians.", ""]
    out: dict = {}
    if a:
        t, d = stall_table(a)
        systems = sorted({r["system"] for r in a})
        best = best_condition(d, systems)
        out["2a"] = d
        out["2a_best"] = best
        locs = sorted({r.get("location") or "?" for r in a})
        by_loc = ""
        if len(locs) > 1:
            lt, ld = stall_table([{**r, "system": f"{r['system']} @ {r.get('location')}", "category": r["category"]}
                                  for r in a])
            out["2a_by_location"] = {k: v for k, v in ld.items() if "·" not in k}
            nl = chr(10)
            by_loc = ("The batch was interrupted and resumed after a move, so its runs come from two networks. "
                      "Stall counts by location (the pooled table above is the primary result):" + nl + nl
                      + nl.join(x for x in lt.splitlines() if "·" not in x) + nl)
        parts += ["### 2a · Append size (20 ms vs 200 ms), trailing room tone at −60 dBFS, 20 repetitions", "", t, "",
                  by_loc,
                  f"Lower stall rate: **{best}** (ties broken by median start lag).", "", "Stalled runs:", "",
                  stalled_list(a), ""]
    if b:
        t, d = stall_table(b)
        systems = sorted({r["system"] for r in b})
        best = best_condition(d, systems)
        out["2b"] = d
        out["2b_best"] = best
        parts += ["### 2b · Trailing audio (room tone −60 dBFS, digital silence, noise bed −50 dBFS), 10 repetitions", "",
                  t, "", f"Lowest stall rate: **{best}**.", "", "Stalled runs:", "", stalled_list(b), ""]
    write_section("exp2ab", "\n".join(parts), out)
    return out


def write_2c() -> dict:
    c = load_rows("exp2c")
    if not c:
        return {}
    recs = condition_summary(c)
    r1 = round1_rows()
    paired = paired_vs_round1(c, r1, "openai")
    lines = ["### 2c · Best stall condition at concurrency 1, all speech categories, 5 repetitions", "",
             markdown_table(recs), "",
             "Per-clip paired difference in end lag at playback vs round-1 OpenAI (median of condition − median of "
             "round 1, same clip; negative = faster):", ""]
    for s, d in paired.items():
        lines.append(f"- {s}: median {d['median_ms']:+,.0f} ms over {d['n_clips']} clips; faster on {d['n_faster']}/"
                     f"{d['n_clips']}. Per clip: " + ", ".join(f"{k.replace('es_en_', '')} {v:+,.0f}"
                                                              for k, v in d["per_clip_ms"].items()))
    lines += ["", "Stalled runs:", "", stalled_list(c)]
    write_section("exp2c", "\n".join(lines), {"summary": recs, "paired_vs_round1": paired})
    return {"summary": recs, "paired": paired}
