"""P1 active-segment trimmer (method to re-implement in the main build).

Inputs: per-frame landmarks (lh, rh: T x 21 x 3; pose: T x 33 x >=2; MediaPipe normalized coords), fps, and the
image aspect w/h. Steps:
  1. x is made pixel-proportional (x * w/h) so distances are isotropic.
  2. Dominant hand = the hand with more wrist motion.
  3. Height h(t) = (shoulder-midline y - wrist y) / shoulder width (higher = raised); a missing hand counts
     as resting. The clip's own resting height = the 10th percentile of h over frames (hands at rest).
  4. Speed s(t) = wrist speed in shoulder widths per second, smoothed over 3 frames.
  5. A frame is "raised" if h > rest + RISE. Onset = first raised frame with s > SPEED; offset = last raised
     frame with s > SPEED. Hysteresis: the segment is then extended while the hand stays raised above
     rest + RISE/2 for frames adjacent to it, at most HYST_S seconds on each side.
  6. Static-hold signs (ASL-LEX path movement "None"): keep at least MIN_HOLD_S after the last fast frame
     while the hand is still raised.
  7. Pad PAD frames each side.
"""
from __future__ import annotations

import numpy as np

RISE = 0.15       # shoulder widths above the clip's resting height (tuned on validation signers)
SPEED = 0.3       # shoulder widths per second (tuned on validation signers)
HYST_S = 0.15     # max extension (s) while the hand stays above rest + RISE/2
MIN_HOLD_S = 0.30
PAD = 4           # frames (tuned on validation signers: pad 2 cost 1.5 points)


def _motion(h):
    c = h[:, 0, :2]
    v = np.diff(c, axis=0)
    ok = ~np.isnan(v).any(1)
    return float(np.abs(v[ok]).sum()) + 0.05 * float((~np.isnan(c).any(1)).sum())


def trim(lh, rh, pose, fps, aspect=1.0, static_hold=False, rise=RISE, speed=SPEED, pad=PAD, hyst_s=HYST_S,
         min_hold_s=MIN_HOLD_S):
    T = len(lh)
    lh, rh, pose = (np.array(a, dtype=np.float32, copy=True) for a in (lh, rh, pose))
    for a in (lh, rh, pose):
        a[..., 0] *= aspect
    dom = lh if _motion(lh) > _motion(rh) else rh
    sh = pose[:, 11:13, :2]
    sw = np.nanmedian(np.linalg.norm(sh[:, 0] - sh[:, 1], axis=-1))
    if not np.isfinite(sw) or sw < 1e-4:
        return 0, T, {"reason": "no shoulders"}
    mid_y = np.nanmedian(sh[..., 1].mean(-1))
    w = dom[:, 0, :2]
    present = ~np.isnan(w).any(1)
    h = np.where(present, (mid_y - w[:, 1]) / sw, -np.inf)
    finite = h[np.isfinite(h)]
    if len(finite) < 3:
        return 0, T, {"reason": "hand rarely visible"}
    rest = float(np.percentile(np.where(np.isfinite(h), h, np.min(finite) - 1), 10))
    rest = max(rest, float(np.min(finite)))
    wf = w.copy()
    for d in range(2):  # fill gaps for speed computation
        col = wf[:, d]
        ok = ~np.isnan(col)
        if ok.sum() >= 2:
            col[~ok] = np.interp(np.where(~ok)[0], np.where(ok)[0], col[ok])
    s = np.r_[0, np.linalg.norm(np.diff(wf, axis=0), axis=1)] * fps / sw
    s = np.convolve(s, np.ones(3) / 3, mode="same")
    s[~present] = 0
    raised = h > rest + rise
    fast = raised & (s > speed)
    idx = np.where(fast)[0]
    if len(idx) == 0:
        return 0, T, {"reason": "no active frames", "rest": rest}
    a, b = int(idx[0]), int(idx[-1])
    lo_ok = h > rest + rise / 2
    k = int(round(hyst_s * fps))
    while a > 0 and lo_ok[a - 1] and idx[0] - a < k:
        a -= 1
    while b < T - 1 and lo_ok[b + 1] and b - idx[-1] < k:
        b += 1
    if static_hold:
        need = int(round(min_hold_s * fps))
        while b < T - 1 and lo_ok[b + 1] and b - idx[-1] < need:
            b += 1
    a, b = max(0, a - pad), min(T, b + 1 + pad)
    return a, b, {"rest": round(rest, 3), "removed_frac": round(1 - (b - a) / T, 3)}
