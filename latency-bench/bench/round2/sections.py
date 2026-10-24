"""Report sections for experiments 3, 4, 5 and 6 (each states its pass criterion)."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from bench.config import RESULTS_DIR
from bench.round2.analyze import load_rows
from bench.round2.common import fmt_ms, med, paired_clip_diffs, round1_rows, wilson, write_section
from bench.round2.summaries import condition_summary, markdown_table, paired_vs_round1, primary_end_lag

MULTI = ("two_utterances", "hesitation", "three_turns")


def _end_values(r: dict, key: str = "end_lag_playback_ms") -> list[float]:
    """Every end lag a run contributes: per utterance for multi-utterance clips, else one."""
    if r.get("utterances") and r["category"] in MULTI:
        return [u[key] for u in r["utterances"] if u.get(key) is not None]
    return [r[key]] if r.get(key) is not None else []


def _pooled(rows: list[dict], system: str) -> dict:
    ok = [r for r in rows if r["system"] == system and r["status"] == "ok"]
    vals = [v for r in ok for v in _end_values(r)]
    cvals = [v for r in ok for v in _end_values(r, "end_lag_playback_catchup125_ms")]
    return {"n_runs": len(ok), "n_clips": len({r["clip_id"] for r in ok}), "n_values": len(vals),
            "catchup_median": med(cvals), "catchup_max": max(cvals) if cvals else None,
            "catchup_over_4s": sum(1 for v in cvals if v > 4000),
            "overlap_s_med": med(r.get("session_overlap_s") for r in ok if r.get("session_overlap_s") is not None),
            "median": med(vals), "max": max(vals) if vals else None, "over_4s": sum(1 for v in vals if v > 4000),
            "stalls": sum(1 for r in ok if r.get("stall")), "chrf": med(r.get("chrf") for r in ok),
            "start": med(r.get("start_lag_ms") for r in ok)}


# ------------------------------------------------------------------ experiment 3
def write_exp3() -> dict:
    rows = load_rows("exp3")
    if not rows:
        return {}
    systems = sorted({r["system"] for r in rows})
    control = next((s for s in systems if "flush-none" in s), None)
    ctrl = _pooled(rows, control) if control else None
    lines = ["## Experiment 3 · Flush on pause and watchdog flush (OpenAI, session rotation)", "",
             "Translation sessions accept no commit event, so a flush is a `session.close` on the active session "
             "(its remaining translation keeps draining into the output) while new source audio goes to a pre-opened, "
             "pre-configured standby session. A causal silero VAD on the sent stream (no look-ahead) triggers "
             "`on_pause` flushes; `watchdog` flushes only if no audible translation arrives within 1,200 ms of the "
             "VAD speech end (min silence 400 ms). A probe showed an idle configured session survives 90 s without "
             "audio and then translates normally, so standby sessions get no keepalive audio. Multi-utterance clips "
             "are scored per utterance (VAD, min silence 300 ms). During a rotation two sessions stream at once; "
             "each session is played on its own track and the tracks are mixed (concatenating two real-time streams "
             "in arrival order, as first specified, would double the audio per second and inflate lags by 10 s or "
             "more; that variant is kept in the data as `end_lag_playback_concat_ms`).", "",
             markdown_table(condition_summary(rows)), "",
             "**Pass criterion:** median end lag at playback ≤ 2,300 ms, no value over 4,000 ms, and chrF no more "
             "than 3 points below the no-flush control (values pooled over all clips; per utterance on "
             "multi-utterance clips).", "",
             "| Condition | runs/clips | values | median end lag | max | over 4 s | with 1.25x catch-up: median / over 4 s | "
             "stalls | chrF | rotations/run | premature flushes (hesitation) | drain med (ms) | sessions/run | "
             "text-only sessions | audio min | pass (1x) | pass (catch-up) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    out = {"pooled": {}, "control": control}
    for s in systems:
        p = _pooled(rows, s)
        ok = [r for r in rows if r["system"] == s and r["status"] == "ok"]
        rot = med(r.get("rotations", 0) for r in ok)
        hes = [r for r in ok if r["category"] == "hesitation"]
        prem_h = sum(r.get("premature_flushes", 0) or 0 for r in hes)
        drains = [d for r in ok for d in (r.get("drain_ms") or [])]
        sess = med(r.get("sessions", 0) for r in ok)
        amin = sum(r.get("session_audio_s", 0) for r in ok) / 60
        chrf_ok = ctrl is None or p["chrf"] is None or ctrl["chrf"] is None or p["chrf"] >= ctrl["chrf"] - 3
        passed = (p["median"] is not None and p["median"] <= 2300 and p["over_4s"] == 0 and chrf_ok)
        passed_c = (p["catchup_median"] is not None and p["catchup_median"] <= 2300 and p["catchup_over_4s"] == 0
                    and chrf_ok)
        p.update({"rotations_med": rot, "premature_hesitation": prem_h, "hesitation_runs": len(hes),
                  "drain_med": med(drains), "sessions_med": sess, "audio_min": round(amin, 2), "pass": passed,
                  "chrf_ok": chrf_ok})
        out["pooled"][s] = p
        silent = sum(r.get("sessions_text_without_speech", 0) or 0 for r in ok)
        p["sessions_text_without_speech"] = silent
        lines.append(f"| {s} | {p['n_runs']}/{p['n_clips']} | {p['n_values']} | {fmt_ms(p['median'])} | "
                     f"{fmt_ms(p['max'])} | {p['over_4s']} | {fmt_ms(p['catchup_median'])} / {p['catchup_over_4s']} | "
                     f"{p['stalls']} | {p['chrf'] if p['chrf'] is None else round(p['chrf'])} | "
                     f"{rot} | {prem_h}/{len(hes)} runs | {fmt_ms(med(drains))} | {sess} | {silent} | {amin:.1f} | "
                     f"{'PASS' if passed else 'fail'} | {'PASS' if passed_c else 'fail'} |")
        p["pass_catchup"] = passed_c
    # premature flushes and accuracy
    lines += ["", "Premature flushes (a flush triggered inside a known sentence span) and accuracy:", ""]
    for s in systems:
        ok = [r for r in rows if r["system"] == s and r["status"] == "ok" and r.get("flushes_checked")]
        if not ok:
            continue
        with_p = [r.get("chrf") for r in ok if r.get("premature_flushes")]
        without = [r.get("chrf") for r in ok if not r.get("premature_flushes")]
        lines.append(f"- {s}: {sum(r['premature_flushes'] for r in ok)} premature of "
                     f"{sum(r['flushes_checked'] for r in ok)} checked flushes; chrF median with a premature flush "
                     f"{med(with_p) if with_p else '—'} (n = {len(with_p)}) vs without {med(without)} (n = {len(without)}).")
    r1 = round1_rows()
    paired = paired_vs_round1(rows, r1, "openai", categories=("medium", "noisy", "long"))
    lines += ["", "Per-clip paired difference vs round-1 OpenAI (same clips; medium, noisy, long):", ""]
    for s, d in paired.items():
        lines.append(f"- {s}: median {d['median_ms']:+,.0f} ms over {d['n_clips']} clips "
                     f"(faster on {d['n_faster']}).")
    out["paired_vs_round1"] = paired
    write_section("exp3", "\n".join(lines), out)
    return out


# ------------------------------------------------------------------ experiment 4
def write_exp4(blocked_reason: str | None = None) -> dict:
    rows = load_rows("exp4")
    if blocked_reason and not rows:
        write_section("exp4", "## Experiment 4 · Gemini Live Translate baseline\n\n**Blocked.** " + blocked_reason,
                      {"blocked": blocked_reason})
        return {"blocked": blocked_reason}
    if not rows:
        return {}
    recs = condition_summary(rows)
    lines = ["## Experiment 4 · Gemini Live Translate baseline", "",
             "Gemini and an OpenAI control (round-1 configuration) ran in one randomized batch so network conditions "
             "match.", "", markdown_table(recs), ""]
    systems = sorted({r["system"] for r in rows})
    gem = [s for s in systems if s.startswith("gemini")]
    ctl = [s for s in systems if s.startswith("openai")]
    if gem and ctl:
        g = [r for r in rows if r["system"] == gem[0] and r["status"] == "ok"]
        o = [r for r in rows if r["system"] == ctl[0] and r["status"] == "ok"]
        d = paired_clip_diffs(g, o, "end_lag_playback_ms")
        lines += ["Per-clip paired difference, Gemini − OpenAI (end lag at playback, ms): "
                  + ", ".join(f"{k.replace('es_en_', '')} {v:+,.0f}" for k, v in d.items()), ""]
    for cat in ("same_language", "silence"):
        lines.append(f"**{cat}:**")
        for s in systems:
            rs = [r for r in rows if r["system"] == s and r["category"] == cat and r["status"] == "ok"]
            if rs:
                lines.append(f"- {s}: output in {sum(1 for r in rs if r.get('produced_output'))}/{len(rs)} runs. "
                             + "; ".join((r.get("provider_output_text") or "")[:60] for r in rs if r.get("produced_output")))
        lines.append("")
    write_section("exp4", "\n".join(lines), {"summary": recs})
    return {"summary": recs}


# ------------------------------------------------------------------ experiment 5
def write_exp5() -> dict:
    a, b = load_rows("exp5a"), load_rows("exp5b")
    if not a and not b:
        return {}
    r1 = round1_rows()
    lines = ["## Experiment 5 · Cascade output pace and single speech context", "",
             "Phrase-by-phrase cascade. *Baseline* = round-1 configuration (one Cartesia context per fragment, "
             "1,000 ms buffer). *single* = one context per turn: fragments are sent with `continue: true`, the tail "
             "with `continue: false`; `mbd` = `max_buffer_delay_ms`; *flush* = Cartesia's default 3,000 ms buffer "
             "with `flush: true` after each fragment. *concise* = interpreter prompt asking for natural, concise "
             "spoken English (~10% fewer syllables, nothing dropped). *adaptive* = `generation_config.speed` 1.2 "
             "when more than 500 ms of audio is queued (1.3 above 1,000 ms), back to 1.0 under 250 ms.", ""]
    out = {}
    for name, rows in (("5a", a), ("5b", b)):
        if not rows:
            continue
        recs = condition_summary(rows)
        out[name] = recs
        lines += [f"### {name}", "", markdown_table(recs), ""]
        paired = paired_vs_round1(rows, r1, "cascade[incremental]")
        out[name + "_paired"] = paired
        lines += ["Per-clip paired difference vs round-1 phrase cascade (end lag at playback):", ""]
        for s, d in paired.items():
            lines.append(f"- {s}: median {d['median_ms']:+,.0f} ms over {d['n_clips']} clips (faster on {d['n_faster']}).")
        lines.append("")
    allrows = a + b
    lines += ["**Pass criterion:** monologue (long) end lag at playback under 5 s with chrF ≥ 85.", "",
              "| Condition | long: end lag med | long: output/source | long: chrF | pass |", "|---|---|---|---|---|"]
    passes = {}
    for s in sorted({r["system"] for r in allrows}):
        lr = [r for r in allrows if r["system"] == s and r["category"] == "long" and r["status"] == "ok"]
        if not lr:
            continue
        e, ratio, ch = med(r.get("end_lag_playback_ms") for r in lr), med(r.get("ratio_span") for r in lr), med(r.get("chrf") for r in lr)
        ok = e is not None and e < 5000 and ch is not None and ch >= 85
        passes[s] = {"end": e, "ratio": ratio, "chrf": ch, "pass": ok}
        rs = "—" if ratio is None else f"{ratio:.2f}"
        cs = "—" if ch is None else f"{ch:.0f}"
        lines.append(f"| {s} | {fmt_ms(e)} | {rs} | {cs} | {'PASS' if ok else 'fail'} |")
    lines += ["", "Adaptive synthesis speed changes per run (median): "
              + ", ".join(f"{s} {med(r.get('speed_changes', 0) for r in allrows if r['system'] == s and r['status'] == 'ok')}"
                          for s in sorted({r["system"] for r in allrows}) if "adaptive" in s)]
    out["long_pass"] = passes
    write_section("exp5", "\n".join(lines), out)
    return out


# ------------------------------------------------------------------ experiment 6
def fragment_llm_times(rows: list[dict]) -> dict[str, dict]:
    """Per-fragment LLM first-token and completion times from raw stage marks."""
    res: dict[str, dict] = {}
    for s in sorted({r["system"] for r in rows}):
        ft, ct = [], []
        for r in rows:
            if r["system"] != s or r["status"] != "ok":
                continue
            p = RESULTS_DIR / "raw" / f"{r['run_id']}.jsonl"
            req, first, done = {}, {}, {}
            for line in open(p, encoding="utf-8"):
                d = json.loads(line)
                if d.get("type") != "event" or d.get("kind") != "stage_mark":
                    continue
                seg = d["meta"].get("seg")
                if d["stage"] == "llm.request" and not d["meta"].get("preset"):
                    req[seg] = d["t_recv_ns"]
                elif d["stage"] == "llm.first_token":
                    first[seg] = d["t_recv_ns"]
                elif d["stage"] == "llm.done":
                    done[seg] = d["t_recv_ns"]
            ft += [(first[k] - req[k]) / 1e6 for k in req if k in first]
            ct += [(done[k] - req[k]) / 1e6 for k in req if k in done]
        res[s] = {"fragments": len(ft), "first_token_med": med(ft), "first_token_p90": float(np.percentile(ft, 90)) if ft else None,
                  "complete_med": med(ct), "complete_p90": float(np.percentile(ct, 90)) if ct else None}
    return res


def write_exp6(blocked: dict[str, str] | None = None, groq_model: str | None = None) -> dict:
    rows = load_rows("exp6")
    lines = ["## Experiment 6 · Translation host and pipelined fragments", ""]
    if blocked:
        lines += ["**Partly blocked:** " + "; ".join(f"{k}: {v}" for k, v in blocked.items()), ""]
    if groq_model:
        lines += [f"Groq model: `{groq_model}`.", ""]
    if not rows:
        write_section("exp6", "\n".join(lines), {"blocked": blocked})
        return {"blocked": blocked}
    recs = condition_summary(rows)
    llm = fragment_llm_times(rows)
    lines += [markdown_table(recs), "", "LLM time per fragment (ms, from request):", "",
              "| Condition | fragments | first token med | p90 | complete med | p90 |", "|---|---|---|---|---|---|"]
    for s, d in llm.items():
        lines.append(f"| {s} | {d['fragments']} | {fmt_ms(d['first_token_med'])} | {fmt_ms(d['first_token_p90'])} | "
                     f"{fmt_ms(d['complete_med'])} | {fmt_ms(d['complete_p90'])} |")
    stage_keys = ("stt_finalize_ms", "translate_wait_ms", "tts_first_byte_ms", "tts_stream_rest_ms")
    lines += ["", "Stage breakdown of the final fragment (medians, ms; sentences = medium + noisy):", "",
              "| Condition | STT finalization | translation wait | TTS first byte | rest of TTS |", "|---|---|---|---|---|"]
    for s in sorted({r["system"] for r in rows}):
        rs = [r for r in rows if r["system"] == s and r["status"] == "ok" and r["category"] in ("medium", "noisy")]
        lines.append(f"| {s} | " + " | ".join(fmt_ms(med(r.get(k) for r in rs)) for k in stage_keys) + " |")
    lines += ["", "**Pass criterion:** sentence (medium + noisy) start lag ≤ 2,000 ms.", ""]
    passes = {}
    for rec in recs:
        if rec["category"] == "sentences":
            ok = rec["start_lag_med"] is not None and rec["start_lag_med"] <= 2000
            passes[rec["system"]] = ok
            lines.append(f"- {rec['system']}: sentence start lag median {fmt_ms(rec['start_lag_med'])} ms "
                         f"{'PASS' if ok else 'fail'}.")
    write_section("exp6", "\n".join(lines), {"summary": recs, "llm": llm, "pass": passes, "blocked": blocked})
    return {"summary": recs, "llm": llm, "pass": passes}
