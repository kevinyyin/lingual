"""Shared helpers for round-2 experiments: statistics, run loading, costs, report sections.

Round-1 artifacts (results/raw, results/audio, results/analysis) are read, never written.
Round-2 outputs live under results/round2/.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from bench.config import RESULTS_DIR
from bench.events import RunData, read_run

R2 = RESULTS_DIR / "round2"
SECTIONS = R2 / "sections"
ROUND1_BATCHES = ("20261013-201319", "20261013-211511")
OPENAI_USD_PER_MIN = 0.034  # docs: gpt-realtime-translate, per audio minute per session


# ------------------------------------------------------------------ statistics
def med(v: Iterable[float | None]) -> float | None:
    a = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.median(a)) if a else None


def p90(v: Iterable[float | None]) -> float | None:
    a = [x for x in v if x is not None]
    return float(np.percentile(a, 90)) if a else None


def boot_ci_median(v: Iterable[float | None], B: int = 5000, seed: int = 0) -> tuple[float, float] | None:
    a = np.array([x for x in v if x is not None], dtype=float)
    if len(a) < 3:
        return None
    rng = np.random.default_rng(seed)
    m = np.median(rng.choice(a, (B, len(a))), axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def fmt_ms(x: float | None) -> str:
    return "—" if x is None else f"{x:,.0f}"


def fmt_ci(ci: tuple[float, float] | None) -> str:
    return "—" if ci is None else f"[{ci[0]:,.0f}–{ci[1]:,.0f}]"


# ------------------------------------------------------------------ runs
def raw_paths(batches: Iterable[str]) -> list[Path]:
    out = []
    for b in batches:
        out += sorted((RESULTS_DIR / "raw").glob(f"{b}_*.jsonl"))
    return out


def load_pcm(run: RunData) -> np.ndarray | None:
    if run.footer and run.footer.output_audio_file:
        return sf.read(str(RESULTS_DIR / "audio" / run.footer.output_audio_file), dtype="int16")[0]
    return None


def system_label(run: RunData) -> str:
    h = run.header
    if h.condition:
        return h.condition
    if h.provider == "cascade":
        return f"cascade[{h.provider_config.get('translate_mode', 'on_turn_end')}]"
    return h.provider


def round1_rows() -> list[dict[str, Any]]:
    """Round-1 per-run metrics as analyzed for the round-1 paper (read-only)."""
    p = RESULTS_DIR / "analysis" / "runs.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def paired_clip_diffs(rows: list[dict], baseline: list[dict], metric: str, label_key: str = "system") -> dict:
    """Per-clip difference of medians (condition - baseline) for clips present in both."""
    by = defaultdict(list)
    for r in rows:
        if r.get(metric) is not None:
            by[r["clip_id"]].append(r[metric])
    base = defaultdict(list)
    for r in baseline:
        if r.get(metric) is not None:
            base[r["clip_id"]].append(r[metric])
    return {c: round(float(np.median(by[c]) - np.median(base[c])), 1) for c in sorted(by) if c in base}


# ------------------------------------------------------------------ costs
def log_cost(batch_id: str, experiment: str, note: str, sessions: int, audio_min_openai: float,
             session_wall_min_openai: float, extra: dict | None = None) -> dict:
    rec = {
        "batch_id": batch_id, "experiment": experiment, "note": note, "openai_sessions": sessions,
        "openai_audio_min": round(audio_min_openai, 3), "openai_session_wall_min": round(session_wall_min_openai, 3),
        "openai_est_usd": round(audio_min_openai * OPENAI_USD_PER_MIN, 3),
        "openai_est_usd_if_billed_by_wall_time": round(session_wall_min_openai * OPENAI_USD_PER_MIN, 3),
        **(extra or {}),
    }
    R2.mkdir(parents=True, exist_ok=True)
    with open(R2 / "costs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def write_section(name: str, markdown: str, data: dict | None = None) -> Path:
    SECTIONS.mkdir(parents=True, exist_ok=True)
    p = SECTIONS / f"{name}.md"
    p.write_text(markdown.strip() + "\n", encoding="utf-8")
    if data is not None:
        (SECTIONS / f"{name}.json").write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    return p
