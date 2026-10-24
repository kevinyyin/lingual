"""`bench analyze`: raw events -> results/analysis/runs.jsonl -> results/summary.csv."""

from __future__ import annotations

import asyncio
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from bench.analysis.metrics import (
    STAGE_COMPONENTS, cascade_stages, describe, evs_values, latency_metrics, output_chunks,
    playback_time_of_output_sample, precise_time_of_sample, provider_texts, simulate_playback,
)
from bench.analysis.quality import chrf, judge_adequacy
from bench.audio import load_wav
from bench.config import RESULTS_DIR, BenchConfig
from bench.corpus import load_index
from bench.events import RunData, read_run
from bench.llm import LLMClient

ANALYSIS_DIR = RESULTS_DIR / "analysis"
RUNS_PATH = ANALYSIS_DIR / "runs.jsonl"
SUMMARY_PATH = RESULTS_DIR / "summary.csv"

LATENCY_METRICS = ["setup_ms", "start_lag_ms", "end_lag_arrival_ms", "end_lag_playback_ms",
                   "first_byte_lag_ms", "evs_mean_ms", "evs_median_ms"]
EVS_CATEGORIES = {"short", "medium", "long", "verb_final", "noisy"}
BEHAVIOR_CATEGORIES = {"same_language", "silence", "overlap"}


def provider_label(run: RunData) -> str:
    p = run.header.provider
    if p == "cascade":
        return f"cascade[{run.header.provider_config.get('translate_mode', 'on_turn_end')}]"
    return p


def raw_files(batches: list[str] | None) -> list[Path]:
    files = sorted((RESULTS_DIR / "raw").glob("*.jsonl"))
    if batches:
        files = [f for f in files if any(f.name.startswith(b + "_") for b in batches)]
    return files


def _source_words(cfg: BenchConfig, run: RunData, clip_path: Path, lang: str | None) -> list[tuple[str, int]]:
    from bench.analysis.align import transcribe_words

    x, sr = load_wav(clip_path)
    asr = transcribe_words(cfg, x, sr, lang)
    sent = run.sends.sent_ns
    return [(w, precise_time_of_sample(sent, int(s * sr), sr, run.header.chunk_ms)) for w, s, _e in asr["words"]]


async def analyze_run(cfg: BenchConfig, path: Path, index: dict, asr: bool, evs: bool,
                      aligner: LLMClient | None, judge: LLMClient | None) -> dict[str, Any]:
    run = read_run(path)
    h = run.header
    clip = index.get(h.clip_id)
    row: dict[str, Any] = {
        "run_id": h.run_id, "batch_id": h.batch_id, "provider": h.provider, "provider_label": provider_label(run),
        "clip_id": h.clip_id, "pair": h.pair, "category": h.category, "synthetic": h.synthetic,
        "audio_source": "synthetic" if h.synthetic else "real", "repetition": h.repetition,
        "order_index": h.order_index, "status": run.footer.status if run.footer else "failed",
        "error": (run.footer.error or "").splitlines()[0][:300] if run.footer and run.footer.error else None,
        "output_audio_file": run.footer.output_audio_file if run.footer else None,
        "config_sha": h.config_sha,
    }
    if row["status"] != "ok":
        return row

    pcm, out_sr = None, None
    if run.footer.output_audio_file:
        pcm, out_sr = sf.read(str(RESULTS_DIR / "audio" / run.footer.output_audio_file), dtype="int16")
    chunks = output_chunks(run, pcm, cfg.audio.output_gate_dbfs)
    row.update(latency_metrics(run, chunks))
    row.update(provider_texts(run))
    row.update(cascade_stages(run, chunks))
    row["n_errors"] = sum(1 for e in run.events if e.kind == "error")

    # ---- ASR of output (what a listener hears) and EVS ----
    row["asr_output_text"] = None
    if asr and pcm is not None and len(pcm) and row.get("produced_output"):
        from bench.analysis.align import transcribe_words

        out_asr = transcribe_words(cfg, pcm.astype(np.float32) / 32768.0, out_sr, h.target_lang)
        row["asr_output_text"] = out_asr["text"]
        if evs and h.category in EVS_CATEGORIES and clip is not None and aligner is not None:
            spans = simulate_playback(chunks)
            tgt = [(w, playback_time_of_output_sample(chunks, spans, int(s * out_sr))) for w, s, _e in out_asr["words"]]
            tgt = [(w, t) for w, t in tgt if t is not None]
            src = _source_words(cfg, run, clip.path, h.source_lang)
            try:
                pairs = await align_words_cached(cfg, aligner, h.source_lang, h.target_lang,
                                                 [w for w, _ in src], [w for w, _ in tgt])
                vals = evs_values([t for _, t in src], [t for _, t in tgt], pairs)
                row["evs_pairs"] = len(vals)
                row["evs_mean_ms"] = round(float(np.mean(vals)), 1) if vals else None
                row["evs_median_ms"] = round(float(np.median(vals)), 1) if vals else None
            except Exception as e:
                row["evs_error"] = f"{type(e).__name__}: {e}"[:200]

    # ---- quality sanity ----
    hyp = row.get("asr_output_text") or row.get("provider_output_text")
    if clip is not None and clip.reference and hyp and h.category not in ("silence",):
        row["chrf"] = chrf(hyp, clip.reference)
        if judge is not None:
            try:
                row["judge_adequacy"] = await judge_adequacy(judge, clip.text or "", clip.reference, hyp, h.target_lang)
            except Exception as e:
                row["judge_error"] = str(e)[:200]
    return row


async def align_words_cached(cfg, aligner, src_lang, tgt_lang, src_words, tgt_words):
    from bench.analysis.align import align_words

    return await align_words(cfg, aligner, src_lang, tgt_lang, src_words, tgt_words)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["provider_label"], r["pair"], r["category"], r["audio_source"])].append(r)
    out = []
    for (prov, pair, cat, src), rs in sorted(groups.items()):
        ok = [r for r in rs if r["status"] == "ok"]
        failed = [r for r in rs if r["status"] == "failed"]
        uns = [r for r in rs if r["status"] == "unsupported"]
        attempted = len(ok) + len(failed)
        row: dict[str, Any] = {
            "provider": prov, "pair": pair, "category": cat, "audio_source": src,
            "n_runs": len(rs), "n_ok": len(ok), "n_failed": len(failed), "n_unsupported": len(uns),
            "failure_rate": round(len(failed) / attempted, 3) if attempted else None,
            "produced_output_rate": round(np.mean([bool(r.get("produced_output")) for r in ok]), 3) if ok else None,
        }
        for m in LATENCY_METRICS + STAGE_COMPONENTS + ["llm_first_token_ms", "tts_first_byte_raw_ms"]:
            d = describe([r.get(m) for r in ok])
            if d["count"] == 0 and m in STAGE_COMPONENTS + ["llm_first_token_ms", "tts_first_byte_raw_ms"]:
                continue
            for k, v in d.items():
                row[f"{m}_{k}"] = v
        ch = [r.get("chrf") for r in ok if r.get("chrf") is not None]
        row["chrf_median"] = round(float(np.median(ch)), 2) if ch else None
        jd = [r.get("judge_adequacy") for r in ok if r.get("judge_adequacy") is not None]
        row["judge_adequacy_median"] = float(np.median(jd)) if jd else None
        out.append(row)
    return out


def write_summary(summary: list[dict[str, Any]], path: Path = SUMMARY_PATH) -> None:
    cols: list[str] = []
    for r in summary:
        for k in r:
            if k not in cols:
                cols.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in summary:
            w.writerow(r)


def load_rows(path: Path = RUNS_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def analyze_all(cfg: BenchConfig, batches: list[str] | None = None, evs: bool = True, asr: bool = True,
                log=print) -> list[dict[str, Any]]:
    files = raw_files(batches)
    if not files:
        log("no raw runs found in results/raw/")
        return []
    index = load_index()
    aligner = judge = None
    if evs and cfg.analysis.evs.enabled:
        aligner = LLMClient(cfg.llm_cfg(cfg.analysis.evs.llm))
        if not aligner.available:
            log(f"[analyze] EVS disabled: {aligner.cfg.api_key_env} not set")
            aligner = None
    if cfg.analysis.quality.llm_judge:
        judge = LLMClient(cfg.llm_cfg(cfg.analysis.quality.judge_llm))
        if not judge.available:
            judge = None

    async def go() -> list[dict[str, Any]]:
        rows = []
        for i, f in enumerate(files):
            try:
                rows.append(await analyze_run(cfg, f, index, asr, evs, aligner, judge))
            except Exception as e:
                log(f"[analyze] {f.name}: {type(e).__name__}: {e}")
            if (i + 1) % 10 == 0 or i + 1 == len(files):
                log(f"[analyze] {i + 1}/{len(files)}")
        for c in (aligner, judge):
            if c is not None:
                await c.aclose()
        return rows

    rows = asyncio.run(go())
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUNS_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    write_summary(summarize(rows))
    log(f"[analyze] wrote {RUNS_PATH} ({len(rows)} runs) and {SUMMARY_PATH}")
    return rows
