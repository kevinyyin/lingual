"""Shared helpers for round 2 (early research spike; not product code).

Reuses round-1 code from ../spike (feature pipeline, model, stats) without modifying it.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SPIKE1 = ROOT / "spike"
R1 = ROOT / "results"
R2 = ROOT / "results2"
CACHE2 = ROOT / "cache" / "s2r2"
R2.mkdir(exist_ok=True)
CACHE2.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SPIKE1))

SEEDS = (0, 1, 2)


def jdump(obj, name):
    with open(R2 / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=_default)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def vocab():
    return json.load(open(ROOT / "vocab.json", encoding="utf-8"))


def citizen_class_map() -> dict[str, str]:
    """ASL Citizen gloss -> vocabulary class (same as round 1)."""
    return {g: e["gloss"] for e in vocab() for g in e["asl_citizen_class_videos"]}


def citizen_rows():
    import pandas as pd
    sp = ROOT / "data/asl-citizen/ASL_Citizen/splits"
    rows = pd.concat([pd.read_csv(sp / f"{s}.csv").assign(split=s) for s in ("train", "val", "test")])
    rows = rows.rename(columns={"Participant ID": "signer", "Video file": "video", "ASL-LEX Code": "code"})
    rows["Gloss"] = rows["Gloss"].astype(str).str.strip()
    return rows


def model_classes() -> list[str]:
    import torch
    return torch.load(ROOT / "cache/s2/model.pt")["classes"]


def load_model(path=None):
    import torch
    from s2_model import SignNet
    ck = torch.load(path or ROOT / "cache/s2/model.pt")
    m = SignNet(len(ck["classes"]))
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck["classes"]


def signer_bootstrap(values_by_signer: dict, stat=np.mean, n: int = 2000, seed: int = 0):
    """95% CI by resampling signers. values_by_signer: signer -> list of per-item values (e.g. 0/1 hits);
    the statistic is computed over the pooled items of the resampled signers."""
    keys = [k for k, v in values_by_signer.items() if len(v)]
    if not keys:
        return None
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        pick = rng.choice(len(keys), len(keys), replace=True)
        pooled = np.concatenate([np.asarray(values_by_signer[keys[i]], float) for i in pick])
        bs.append(stat(pooled))
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def acc_with_ci(hits: np.ndarray, signers: np.ndarray, seed: int = 0) -> dict:
    hits = np.asarray(hits, float)
    by = {}
    for h, s in zip(hits, signers):
        by.setdefault(str(s), []).append(h)
    return {"acc": round(float(hits.mean()), 4) if len(hits) else None, "ci": signer_bootstrap(by, seed=seed),
            "n": int(len(hits)), "signers": len(by)}


def pstats(xs):
    xs = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return {"n": 0, "median": None, "p90": None, "max": None}
    a = np.array(xs)
    return {"n": len(xs), "median": round(float(np.median(a)), 2), "p90": round(float(np.percentile(a, 90)), 2),
            "max": round(float(a.max()), 2)}
