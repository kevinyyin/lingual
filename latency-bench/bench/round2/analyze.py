"""Round-2 per-run analysis. Writes results/round2/analysis/<name>.jsonl (never round-1 files).

Adds to the round-1 metrics:
  * stall: start lag > 5 s, or no audible output before the harness began closing the
    session (runner.finish_called); held_until_flush: audible output exists, but only
    after that moment
  * output/source duration ratios (span = onset..offset as in round 1; speech = VAD speech only)
  * per-utterance lags for clips with several VAD speech segments (min silence 300 ms)
  * sessions, audio seconds and session wall time (cost), rotations (experiment 3)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from bench.analysis.metrics import (
    PlaybackPolicy, cascade_stages, latency_metrics, output_chunks, policy_end_lag_ms, provider_texts,
    simulate_playback_policy, simulate_playback_tracks, source_times, utterance_metrics,
)
from bench.analysis.quality import chrf
from bench.config import BenchConfig
from bench.corpus import load_index
from bench.events import RunData, read_run
from bench.round2.common import R2, load_pcm, med, raw_paths, system_label

STALL_START_MS = 5000.0
MULTI_UTT = {"two_utterances", "hesitation", "three_turns"}
CLOSE_MARKS = ("runner.finish_called", "openai.session.close_sent", "gemini.audioStreamEnd_sent", "stt.close_sent")


def close_time(run: RunData) -> int | None:
    for name in CLOSE_MARKS:
        t = next((e.t_recv_ns for e in run.events if e.stage == name), None)
        if t is not None:
            return t
    return None


def analyze_r2_run(cfg: BenchConfig, path: Path, index: dict, asr: bool = True) -> dict[str, Any]:
    run = read_run(path)
    h = run.header
    clip = index.get(h.clip_id)
    row: dict[str, Any] = {
        "run_id": h.run_id, "batch_id": h.batch_id, "experiment": h.experiment, "system": system_label(run),
        "provider": h.provider, "clip_id": h.clip_id, "category": h.category, "repetition": h.repetition,
        "order_index": h.order_index, "chunk_ms": h.chunk_ms, "send_alignment": h.send_alignment,
        "clip_transform": h.clip_transform, "status": run.footer.status if run.footer else "failed",
        "error": (run.footer.error or "").splitlines()[0][:300] if run.footer and run.footer.error else None,
        "output_audio_file": run.footer.output_audio_file if run.footer else None,
        # the first round-2 batch ran before BENCH_LOCATION was set (Philadelphia); later ones record it
        "location": (h.machine or {}).get("location") if (h.machine or {}).get("location") not in (None, "unspecified")
        else "Philadelphia (inferred: before location was recorded)",
        "power_source": (h.machine or {}).get("power_source"),
    }
    sess = [e.meta for e in run.events if e.stage == "cost.session"]
    row["sessions"] = len(sess)
    row["session_audio_s"] = round(sum(s.get("audio_s", 0) for s in sess), 3)
    row["session_wall_s"] = round(sum(s.get("wall_s", 0) for s in sess), 3)
    row["rotations"] = sum(1 for e in run.events if e.stage == "rotation.promote")
    row["flush_triggers"] = [round(e.t_recv_ns / 1e6, 1) for e in run.events if e.stage == "rotation.trigger"]
    row["speed_changes"] = sum(1 for e in run.events if e.stage == "tts.speed_change")
    if row["status"] != "ok":
        return row

    pcm = load_pcm(run)
    chunks = output_chunks(run, pcm, cfg.audio.output_gate_dbfs)
    row.update(latency_metrics(run, chunks))
    row.update(provider_texts(run))
    row.update(cascade_stages(run, chunks))
    blocks = simulate_playback_policy(chunks, PlaybackPolicy("none"))
    _t_on0, t_off0 = source_times(run)
    sess_of = [e.meta.get("session", 0) for e in run.events if e.kind == "audio_out" and e.n_samples]
    catchup = PlaybackPolicy("catchup", 500, 1.25)
    if len(set(sess_of)) > 1 and len(sess_of) == len(chunks):
        # parallel sessions: one playback track per session, mixed (see simulate_playback_tracks)
        row["end_lag_playback_concat_ms"] = row.get("end_lag_playback_ms")  # literal arrival-order merge
        blocks, row["session_overlap_s"] = simulate_playback_tracks(chunks, sess_of)
        row["end_lag_playback_ms"] = policy_end_lag_ms(chunks, blocks, t_off0)
        cblocks, _ = simulate_playback_tracks(chunks, sess_of, catchup)
    else:
        cblocks = simulate_playback_policy(chunks, catchup)
    row["end_lag_playback_catchup125_ms"] = policy_end_lag_ms(chunks, cblocks, t_off0)

    # stall / held-until-flush
    t_close = close_time(run)
    aud = [c for c in chunks if c.first_audible is not None]
    # a translation is expected only for source-language speech: on same_language clips silence is
    # the desired behaviour, so they never count as stalls
    speech = h.source.onset_sample is not None and h.category != "same_language"
    first_aud = aud[0].t_recv_ns if aud else None
    before_close = first_aud is not None and t_close is not None and first_aud < t_close
    row["t_close_rel_offset_ms"] = None
    _t_on, t_off = source_times(run)
    if t_close is not None and t_off is not None:
        row["t_close_rel_offset_ms"] = round((t_close - t_off) / 1e6, 1)
    row["held_until_flush"] = bool(speech and aud and not before_close)
    row["stall"] = bool(speech and ((row.get("start_lag_ms") or 0) > STALL_START_MS or not before_close))

    # duration ratios
    st = h.source
    if speech:
        span_s = (st.offset_sample - st.onset_sample) / st.sample_rate
        speech_s = sum(b - a for a, b in st.segments) / st.sample_rate if st.segments else span_s
        row["ratio_span"] = round(row["audible_output_s"] / span_s, 3) if span_s > 0 else None
        row["ratio_speech"] = round(row["audible_output_s"] / speech_s, 3) if speech_s > 0 else None
        row["source_speech_only_s"] = round(speech_s, 3)

    # per-utterance (multi-utterance clips, and any clip VAD splits)
    if speech and (h.category in MULTI_UTT or len(st.segments) > 1):
        utts = utterance_metrics(run, chunks, blocks)
        cu = utterance_metrics(run, chunks, cblocks)
        for u, c in zip(utts, cu):
            u["end_lag_playback_catchup125_ms"] = c["end_lag_playback_ms"]
        row["utterances"] = utts
        row["utt_end_lag_playback_catchup125_ms"] = med([u["end_lag_playback_catchup125_ms"] for u in utts])
        row["utt_end_lag_playback_ms"] = med([u["end_lag_playback_ms"] for u in utts])  # median over utterances
        row["utt_start_lag_median_ms"] = med([u["start_lag_ms"] for u in utts])

    # per-session outcome: did each session that produced translated TEXT also produce audible SPEECH?
    if row["sessions"] > 1 and pcm is not None:
        thr = int(10 ** (cfg.audio.output_gate_dbfs / 20) * 32768)
        silent = 0
        for sidx in {e.meta.get("session") for e in run.events if e.kind == "text_out_delta"}:
            txt = "".join(e.text or "" for e in run.events if e.kind == "text_out_delta" and e.meta.get("session") == sidx)
            loud = any(e.kind == "audio_out" and e.meta.get("session") == sidx and e.n_samples and
                       np.abs(pcm[e.audio_offset:e.audio_offset + e.n_samples].astype(np.int32)).max(initial=0) > thr
                       for e in run.events)
            if txt.strip() and not loud:
                silent += 1
        row["sessions_text_without_speech"] = silent

    # session rotation (experiment 3): drains and premature flushes
    if row["rotations"]:
        closes = {e.meta.get("session"): e.t_recv_ns for e in run.events if e.stage == "rotation.close_sent"}
        closed = {e.meta.get("session"): e.t_recv_ns for e in run.events if e.stage == "openai.session.closed"}
        row["drain_ms"] = [round((closed[s] - t) / 1e6, 1) for s, t in closes.items() if s in closed]
        row["drained_audio_s"] = round(sum(e.n_samples / (e.sample_rate or 24000) for e in run.events
                                           if e.kind == "audio_out" and e.meta.get("after_promotion")), 3)
        spans = (clip.meta.get("sentence_spans") if clip else None) or (
            [(st.onset_sample, st.offset_sample)] if h.category in ("medium", "noisy") else None)
        if spans and run.sends:
            sent = run.sends.sent_ns
            prem = []
            for e in run.events:
                if e.stage != "rotation.trigger":
                    continue
                k = max(0, int(np.searchsorted(sent, e.t_recv_ns, side="right")) - 1)
                pos = int((k + 1) * h.chunk_ms / 1000 * st.sample_rate)  # audio sent so far, prepared-rate samples
                inside = any(a < pos < b for a, b in spans)
                prem.append(inside)
            row["premature_flushes"] = sum(prem)
            row["flushes_checked"] = len(prem)

    # accuracy screen
    row["asr_output_text"] = None
    if asr and pcm is not None and len(pcm) and row.get("produced_output"):
        from bench.analysis.align import transcribe_words

        row["asr_output_text"] = transcribe_words(cfg, pcm.astype(np.float32) / 32768.0,
                                                  run.footer.output_sample_rate, h.target_lang)["text"]
    hyp = row.get("asr_output_text") or row.get("provider_output_text")
    if clip is not None and clip.reference and hyp and h.category != "silence":
        row["chrf"] = chrf(hyp, clip.reference)
    return row


def analyze_batches(cfg: BenchConfig, name: str, batches: list[str], asr: bool = True, log=print) -> list[dict]:
    index = load_index()
    rows = []
    paths = raw_paths(batches)
    for i, p in enumerate(paths):
        try:
            rows.append(analyze_r2_run(cfg, p, index, asr))
        except Exception as e:  # never lose the batch to one bad file
            log(f"[r2-analyze] {p.name}: {type(e).__name__}: {e}")
        if (i + 1) % 25 == 0:
            log(f"[r2-analyze] {i + 1}/{len(paths)}")
    out = R2 / "analysis" / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    log(f"[r2-analyze] wrote {out} ({len(rows)} runs)")
    return rows


def load_rows(name: str) -> list[dict]:
    p = R2 / "analysis" / f"{name}.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []
