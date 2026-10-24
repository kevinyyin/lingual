"""Raw landmark Frame -> normalized feature vector. Shared by training and runtime.

Layout (FEAT_DIM = 155):
  [0:64)    left hand: 21x3 wrist-relative / shoulder width, then presence flag
  [64:128)  right hand: same
  [128:134) left, right wrist relative to shoulder midpoint / shoulder width
  [134:155) nose, shoulders, elbows, wrists (pose) relative to shoulder midpoint / shoulder width
No blendshapes: the face is read separately by grammar.py and emotion.py.
"""

import numpy as np

from .protocol import Frame

POSE_POINTS = [0, 11, 12, 13, 14, 15, 16]
FEAT_DIM = 2 * 64 + 2 * 3 + len(POSE_POINTS) * 3


def frame_features(f: Frame) -> np.ndarray:
    pose = np.asarray(f.pose, dtype=np.float32)[:, :3]
    mid = (pose[11] + pose[12]) / 2
    width = max(float(np.linalg.norm((pose[11] - pose[12])[:2])), 1e-3)
    out = np.zeros(FEAT_DIM, dtype=np.float32)
    for i, hand in enumerate((f.left, f.right)):
        if hand is None:
            continue
        h = np.asarray(hand, dtype=np.float32)
        out[i * 64:i * 64 + 63] = ((h - h[0]) / width).ravel()
        out[i * 64 + 63] = 1.0
        out[128 + i * 3:131 + i * 3] = (h[0] - mid) / width
    out[134:] = ((pose[POSE_POINTS] - mid) / width).ravel()
    return out


def sequence_features(frames: list[Frame], T: int = 32) -> np.ndarray:
    return resample(np.stack([frame_features(f) for f in frames]), T)


def resample(x: np.ndarray, T: int = 32) -> np.ndarray:
    """(n, F) per-frame features -> (T, F), linearly between frames."""
    src = np.linspace(0, len(x) - 1, T)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, len(x) - 1)
    w = (src - lo)[:, None].astype(np.float32)
    return x[lo] * (1 - w) + x[hi] * w
