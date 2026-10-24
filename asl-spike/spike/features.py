"""Pose features for isolated-sign recognition (S2), identical for ASL Citizen videos (our MediaPipe
extraction) and Kaggle asl-signs landmark files (MediaPipe Holistic, by the competition hosts).

Per frame: both hands (21 x,y each), 8 upper-body pose points, 6 face points; all centered on the
shoulder midpoint and scaled by shoulder width. The dominant hand (the one that moves more over the clip)
is always put in the "right" slot, mirroring x when needed, so left-handed signers and left/right label
conventions do not matter. Clips are trimmed to the frames where a hand is visible (with a margin) and
resampled to T frames.
"""
from __future__ import annotations

import numpy as np

T = 32
POSE_IDX = [0, 11, 12, 13, 14, 15, 16, 23]          # nose, shoulders, elbows, wrists, left hip
FACE_KAGGLE_IDX = [1, 13, 14, 105, 334, 152]        # nose tip, upper/lower lip, brows, chin (468 mesh)
# position of the same points inside extract.FACE_SUBSET
FACE_SUBSET_POS = [8, 1, 2, 11, 14, 20]
# mirror map for pose points when swapping sides
POSE_MIRROR = [0, 2, 1, 4, 3, 6, 5, 7]
N_FEAT = 2 * 21 * 2 + len(POSE_IDX) * 2 + len(FACE_KAGGLE_IDX) * 2 + 2   # + 2 hand-present flags


def from_holistic(d: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """extract.holistic_video output -> (lh, rh, pose8, face6), each (F, n, 2)."""
    return (d["lh"][..., :2], d["rh"][..., :2], d["pose"][:, POSE_IDX, :2], d["face"][:, FACE_SUBSET_POS, :2])


def from_kaggle(df) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frames = np.sort(df["frame"].unique())
    fi = {f: i for i, f in enumerate(frames)}
    F = len(frames)

    def grab(typ, n, idx=None):
        out = np.full((F, n, 2), np.nan, np.float32)
        sub = df[df["type"] == typ]
        if idx is not None:
            sub = sub[sub["landmark_index"].isin(idx)]
            remap = {k: j for j, k in enumerate(idx)}
            li = sub["landmark_index"].map(remap).to_numpy()
        else:
            li = sub["landmark_index"].to_numpy()
        out[sub["frame"].map(fi).to_numpy(), li] = sub[["x", "y"]].to_numpy(np.float32)
        return out

    return grab("left_hand", 21), grab("right_hand", 21), grab("pose", len(POSE_IDX), POSE_IDX), \
        grab("face", len(FACE_KAGGLE_IDX), FACE_KAGGLE_IDX)


def _motion(h: np.ndarray) -> float:
    c = np.nanmean(h, axis=1)
    v = np.diff(c, axis=0)
    ok = ~np.isnan(v).any(1)
    return float(np.abs(v[ok]).sum()) + 0.05 * float((~np.isnan(c).any(1)).sum())


def normalize(lh, rh, pose, face, t: int = T, return_meta: bool = False):
    lh, rh, pose, face = (a.astype(np.float32).copy() for a in (lh, rh, pose, face))
    # dominant hand -> right slot; mirror x around the body midline when swapping
    swap = _motion(lh) > _motion(rh)
    if swap:
        lh, rh = rh, lh
        for a in (lh, rh, pose, face):
            a[..., 0] = 1.0 - a[..., 0]
        pose = pose[:, POSE_MIRROR]
    # center/scale from shoulders (median over the clip)
    sh = pose[:, 1:3]
    mid = np.nanmedian(sh.mean(1), axis=0)
    width = np.nanmedian(np.linalg.norm(sh[:, 0] - sh[:, 1], axis=-1))
    if not np.isfinite(width) or width < 1e-3:
        mid = np.nanmedian(np.nanmean(np.concatenate([lh, rh], 1), 1), 0)
        width = 0.3
    if not np.all(np.isfinite(mid)):
        mid = np.array([0.5, 0.5], np.float32)
    lh, rh, pose, face = ((a - mid) / width for a in (lh, rh, pose, face))
    # trim to hand-visible frames (+3 frame margin)
    present = ~(np.isnan(lh).all((1, 2)) & np.isnan(rh).all((1, 2)))
    idx = np.where(present)[0]
    lo, hi = (max(0, idx[0] - 3), min(len(present), idx[-1] + 4)) if len(idx) else (0, len(present))
    seq = [a[lo:hi] for a in (lh, rh, pose, face)]
    F = hi - lo
    flags = np.stack([~np.isnan(seq[0]).all((1, 2)), ~np.isnan(seq[1]).all((1, 2))], 1).astype(np.float32)
    x = np.concatenate([seq[0].reshape(F, -1), seq[1].reshape(F, -1), seq[2].reshape(F, -1), seq[3].reshape(F, -1),
                        flags], 1)
    x = np.nan_to_num(x, nan=0.0)
    # resample to t frames (linear interpolation in time)
    if F == 0:
        x = np.zeros((t, N_FEAT), np.float32)
    else:
        src = np.linspace(0, F - 1, t)
        i0 = np.floor(src).astype(int)
        i1 = np.minimum(i0 + 1, F - 1)
        w = (src - i0)[:, None]
        x = (x[i0] * (1 - w) + x[i1] * w).astype(np.float32)
    if return_meta:
        return x, {"swapped": bool(swap), "frames_used": int(F), "hand_frames": int(present.sum())}
    return x
