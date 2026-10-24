"""Per-clip hand statistics used by R1 (PopSign gap) - computed from raw landmarks, both datasets alike.

both_frac      frames with both hands detected / frames with at least one hand detected
dom_side       which hand moved more ("left"/"right" label as stored)
dom_loc        dominant wrist mean position relative to the nose, in shoulder widths (x: towards the
               signer's dominant side is not normalized - raw image x; y: down positive)
dom_flex       mean PIP flexion angle (deg) over index..pinky of the dominant hand (0 = straight)
"""
from __future__ import annotations

import numpy as np

FINGERS = [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]  # MCP, PIP, TIP


def _motion(h):
    c = np.nanmean(h[..., :2], axis=1)
    v = np.diff(c, axis=0)
    ok = ~np.isnan(v).any(1)
    return float(np.abs(v[ok]).sum()) + 0.05 * float((~np.isnan(c).any(1)).sum())


def flexion(h):
    """mean PIP angle per frame (deg), NaN when the hand is missing. h: (T,21,3)."""
    out = []
    for a, b, c in FINGERS:
        v1 = h[:, b] - h[:, a]
        v2 = h[:, c] - h[:, b]
        cos = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9)
        out.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return np.nanmean(np.stack(out, 1), 1)


def hand_stats(lh, rh, pose) -> dict:
    """lh/rh: (T,21,3) normalized image coords with NaN for missing; pose: (T,33,>=2)."""
    l_ok = ~np.isnan(lh[:, 0, 0])
    r_ok = ~np.isnan(rh[:, 0, 0])
    anyh = l_ok | r_ok
    both = l_ok & r_ok
    dom_is_left = _motion(lh) > _motion(rh)
    dom = lh if dom_is_left else rh
    nose = pose[:, 0, :2]
    sw = np.nanmedian(np.linalg.norm(pose[:, 11, :2] - pose[:, 12, :2], axis=-1))
    rel = (dom[:, 0, :2] - nose) / (sw if np.isfinite(sw) and sw > 1e-3 else np.nan)
    with np.errstate(all="ignore"):
        fl = flexion(dom)
        return {"frames": int(len(lh)), "hand_frames": int(anyh.sum()),
                "both_frac": float(both.sum() / max(anyh.sum(), 1)),
                "dom_side": "left" if dom_is_left else "right",
                "dom_loc_x": float(np.nanmean(np.abs(rel[:, 0]))) if np.isfinite(rel[:, 0]).any() else None,
                "dom_loc_y": float(np.nanmean(rel[:, 1])) if np.isfinite(rel[:, 1]).any() else None,
                "dom_flex": float(np.nanmean(fl)) if np.isfinite(fl).any() else None}


def kaggle_arrays(df):
    frames = np.sort(df["frame"].unique())
    fi = {f: i for i, f in enumerate(frames)}
    F = len(frames)

    def grab(typ, n):
        out = np.full((F, n, 3), np.nan, np.float32)
        sub = df[df["type"] == typ]
        out[sub["frame"].map(fi).to_numpy(), sub["landmark_index"].to_numpy()] = sub[["x", "y", "z"]].to_numpy(np.float32)
        return out
    return grab("left_hand", 21), grab("right_hand", 21), grab("pose", 33)
