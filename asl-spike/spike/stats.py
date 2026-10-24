"""Small stats helpers: medians with p90/max, Wilson intervals, bootstrap CIs (seeded)."""
from __future__ import annotations

import math

import numpy as np


def pstats(xs) -> dict:
    xs = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return {"n": 0, "median": None, "p90": None, "max": None}
    a = np.array(xs)
    return {"n": len(xs), "median": round(float(np.median(a)), 1), "p90": round(float(np.percentile(a, 90)), 1),
            "max": round(float(a.max()), 1), "min": round(float(a.min()), 1)}


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def boot_ci(xs, stat=np.mean, n: int = 5000, seed: int = 0) -> list[float] | None:
    xs = np.asarray(xs, float)
    if len(xs) == 0:
        return None
    rng = np.random.default_rng(seed)
    bs = [stat(rng.choice(xs, len(xs))) for _ in range(n)]
    return [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
