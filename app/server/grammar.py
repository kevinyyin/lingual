"""Facial grammar over a whole signed sentence: yes/no question (brows up), wh-question (brows down), negation (headshake).

Brows are measured against the signer's calibrated resting face. Emotion is read separately (emotion.py) and never here.
"""

import numpy as np

from .protocol import Frame
from .session import Baseline, Grammar

UP_T = 0.15  # brow raise above baseline -> yes/no question; tuned in plan Task 18
DOWN_T = 0.15  # brow lowering above baseline -> wh-question
SHAKE_REVERSALS = 2  # head yaw direction changes
SHAKE_P2P = 8.0  # degrees, peak to peak
HYSTERESIS = 3.0  # degrees a reversal must travel to count
MIN_FACE = 0.3  # fraction of frames with a face, else no reading


def _middle(frames: list[Frame]) -> list[Frame]:
    cut = int(len(frames) * 0.1)
    return frames[cut:len(frames) - cut] or frames


def _headshake(yaw: np.ndarray) -> bool:
    """Counts yaw direction reversals that travel at least HYSTERESIS degrees (a zigzag counter)."""
    if len(yaw) < 5 or np.ptp(yaw) < SHAKE_P2P:
        return False
    y = np.convolve(yaw, np.ones(3) / 3, mode="valid")
    direction, extreme, reversals = 0, y[0], 0
    for v in y[1:]:
        if direction == 0:
            if abs(v - extreme) >= HYSTERESIS:
                direction, extreme = (1 if v > extreme else -1), v
        elif (v - extreme) * direction > 0:
            extreme = v  # still turning the same way
        elif abs(v - extreme) >= HYSTERESIS:
            direction, extreme, reversals = -direction, v, reversals + 1
    return reversals >= SHAKE_REVERSALS


def read_grammar(frames: list[Frame], baseline: Baseline) -> Grammar:
    mid = _middle(frames)
    faces = [f for f in mid if f.blend is not None]
    if not mid or len(faces) < MIN_FACE * len(mid):
        return {"question": None, "negation": False}

    def mean(*keys: str) -> float:
        return float(np.mean([np.mean([f.blend.get(k, 0.0) for k in keys]) for f in faces]))

    def base(*keys: str) -> float:
        return float(np.mean([baseline.brow.get(k, 0.0) for k in keys]))

    up = (mean("browInnerUp") + mean("browOuterUpLeft", "browOuterUpRight")) / 2 \
        - (base("browInnerUp") + base("browOuterUpLeft", "browOuterUpRight")) / 2
    down = mean("browDownLeft", "browDownRight") - base("browDownLeft", "browDownRight")
    question = "yn" if up > UP_T else "wh" if down > DOWN_T else None
    yaw = np.array([f.head[0] for f in mid if f.head is not None])
    return {"question": question, "negation": _headshake(yaw)}
