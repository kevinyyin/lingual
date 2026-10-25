"""Per-clip motion and condition features, to explain wrong reads (step 3).

Image landmarks with x corrected for the video's aspect (x * w/h), distances in shoulder widths (sw), speeds per second,
over the active span (frames with the dominant hand raised above the hips). Hands are re-assigned to the nearer pose
wrist (15: signer's left, 16: signer's right): MediaPipe's own left/right labels are wrong on a fifth of frames, and how
often they disagree is itself a feature.
"""
import numpy as np

from server.protocol import Frame

TIPS = (4, 8, 12, 16, 20)
PIP = ((5, 6, 7), (9, 10, 11), (13, 14, 15), (17, 18, 19))  # knuckle, middle joint, end joint of each finger


def _xy(p, aspect):
    return np.array([p[0] * aspect, p[1]], float)


def _hand(h, aspect):
    return None if h is None else np.array([[q[0] * aspect, q[1], q[2] * aspect] for q in h], float)


def _sides(f: Frame, aspect: float):
    """(signer-left hand, signer-right hand, label disagreed) by nearness of each hand's wrist to pose wrists 15/16."""
    w15, w16 = _xy(f.pose[15], aspect), _xy(f.pose[16], aspect)
    lab = {"left": _hand(f.left, aspect), "right": _hand(f.right, aspect)}
    out, flips = {15: None, 16: None}, 0
    for label, h in lab.items():
        if h is None:
            continue
        side = 15 if np.linalg.norm(h[0, :2] - w15) <= np.linalg.norm(h[0, :2] - w16) else 16
        if out[side] is None:
            out[side] = h
        # MediaPipe's "Left" is the signer's right on an unmirrored camera: count a disagreement either way round
        flips += int((label == "left") != (side == 16))
    return out[15], out[16], flips


def _angles_back(h: np.ndarray, right: bool) -> int:
    """Fingers whose middle joint bends backward past 45 degrees (tracker garbage), as in the avatar's hand model."""
    F = h[9] - h[0]
    F /= np.linalg.norm(F) + 1e-9
    S = h[5] - h[17]
    S = S - F * (S @ F)
    S /= np.linalg.norm(S) + 1e-9
    n = 0
    for m, p, d in PIP:
        r = h[p] - h[m]
        r /= np.linalg.norm(r) + 1e-9
        dd = h[d] - h[p]
        dd /= np.linalg.norm(dd) + 1e-9
        hh = S - r * (S @ r)
        hh /= np.linalg.norm(hh) + 1e-9
        pp = (1 if right else -1) * np.cross(hh, r)
        n += np.degrees(np.arctan2(dd @ pp, dd @ r)) < -45
    return n


def features(frames: list[Frame], meta: dict) -> dict:
    aspect = (meta.get("w") or 4) / (meta.get("h") or 3)
    fps = meta.get("fps") or 30.0
    if not frames:
        return {"empty": 1}
    ts = np.array([f.t for f in frames]) / 1000.0
    sh = np.array([[_xy(f.pose[11], aspect), _xy(f.pose[12], aspect)] for f in frames])
    sw = float(np.median(np.linalg.norm(sh[:, 0] - sh[:, 1], axis=1))) or 0.2
    mid = sh.mean(1)
    hips_y = np.array([(f.pose[23][1] + f.pose[24][1]) / 2 for f in frames])
    nose = np.array([_xy(f.pose[0], aspect) for f in frames])
    mouth = np.array([(_xy(f.pose[9], aspect) + _xy(f.pose[10], aspect)) / 2 for f in frames])
    chin = mouth + (mouth - nose) * 0.8
    face_lo = np.array([np.min([_xy(f.pose[i], aspect) for i in range(11)], 0) for f in frames])
    face_hi = np.array([np.max([_xy(f.pose[i], aspect) for i in range(11)], 0) for f in frames])
    pad = 0.15 * sw
    sides = [_sides(f, aspect) for f in frames]
    L = [s[0] for s in sides]
    R = [s[1] for s in sides]

    def path(hs):
        pts = [h[0, :2] if h is not None else None for h in hs]
        return sum(np.linalg.norm(a - b) for a, b in zip(pts, pts[1:]) if a is not None and b is not None)

    dom_right = path(R) >= path(L)  # the hand that moves more
    D, O = (R, L) if dom_right else (L, R)
    raised = np.array([(d is not None and d[0, 1] < hips_y[i] - 0.05) for i, d in enumerate(D)])
    idx = np.where(raised)[0]
    if len(idx) < 2:
        idx = np.array([i for i, d in enumerate(D) if d is not None])
    out: dict = {"left_handed": int(not dom_right), "fps": fps, "height_px": meta.get("h"), "frames": len(frames)}
    if len(idx) < 2:
        out.update({"active_frames": int(len(idx)), "hand_detect_frac": 0.0})
        return out
    a0, a1 = idx[0], idx[-1]
    span = range(a0, a1 + 1)
    dur = max(ts[a1] - ts[a0], 1e-3)
    det = np.array([D[i] is not None for i in span])
    out.update({"t_active0": float(frames[a0].t), "t_active1": float(frames[a1].t),
                "active_frames": len(span), "duration_s": float(dur), "hand_detect_frac": float(det.mean()),
                "both_hands_frac": float(np.mean([D[i] is not None and O[i] is not None for i in span])),
                "handedness_flip_frac": float(np.mean([sides[i][2] > 0 for i in span]))})
    # head
    heads = np.array([f.head for f in (frames[i] for i in span) if f.head is not None], float)
    if len(heads) > 2:
        th = np.array([frames[i].t for i in span if frames[i].head is not None]) / 1000.0
        vel = np.abs(np.diff(heads, axis=0)) / np.maximum(np.diff(th)[:, None], 1e-3)
        for k, name in enumerate(("yaw", "pitch", "roll")):
            out[f"head_{name}_range"] = float(np.ptp(heads[:, k]))
        out["head_speed_max"] = float(np.percentile(vel.max(1), 95))
    rel_nose = (nose[list(span)] - mid[list(span)]) / sw
    out["nose_disp"] = float(np.linalg.norm(rel_nose - np.median(rel_nose, 0), axis=1).max())
    sw_t = np.linalg.norm(sh[list(span), 0] - sh[list(span), 1], axis=1)
    out["torso_turn_proxy"] = float(np.ptp(sw_t) / sw)  # a turning torso shortens the shoulders' image width
    out["shrug"] = float(np.ptp(mid[list(span), 1]) / sw)
    # hand location
    dw = np.array([D[i][0, :2] for i in span if D[i] is not None])
    di = [i for i in span if D[i] is not None]
    to = lambda ref: np.linalg.norm(dw - ref[di], axis=1) / sw  # noqa: E731
    out.update({"wrist_nose_min": float(to(nose).min()), "wrist_nose_med": float(np.median(to(nose))),
                "wrist_chin_min": float(to(chin).min()),
                "wrist_chest_med": float(np.median(np.linalg.norm(dw - (mid[di] + [0, 0.5 * sw]), axis=1) / sw))})
    tips_near = [np.min(np.linalg.norm(D[i][list(TIPS), :2] - nose[i], axis=1)) / sw for i in di]
    out["near_face_frac"] = float(np.mean(np.array(tips_near) < 0.5))
    out["over_face_frac"] = float(np.mean([np.any(np.all((D[i][list(TIPS), :2] > face_lo[i] - pad) & (D[i][list(TIPS), :2] < face_hi[i] + pad), axis=1))
                                           for i in di]))
    rel = (np.median(dw, 0) - np.median(mid[di], 0)) / sw  # median wrist relative to the shoulders' midpoint
    out["loc_up"], out["loc_side"] = float(-rel[1]), float(abs(rel[0]))
    out["loc_bin"] = ("head" if -rel[1] > 0.6 else "neck" if -rel[1] > 0.25 else "torso" if -rel[1] > -0.9 else "low")
    # two hands
    both = [i for i in span if D[i] is not None and O[i] is not None]
    if both:
        out["hands_dist_min"] = float(min(np.linalg.norm(D[i][0, :2] - O[i][0, :2]) for i in both) / sw)

        def ov(a, b):
            lo, hi = np.maximum(a[:, :2].min(0), b[:, :2].min(0)), np.minimum(a[:, :2].max(0), b[:, :2].max(0))
            inter = np.prod(np.clip(hi - lo, 0, None))
            small = min(np.prod(np.ptp(a[:, :2], 0)), np.prod(np.ptp(b[:, :2], 0)))
            return inter / small if small > 0 else 0.0

        out["hands_overlap_frac"] = float(np.mean([ov(D[i], O[i]) > 0.15 for i in both]))
        out["hands_cross_frac"] = float(np.mean([(D[i][0, 0] - mid[i, 0]) * (O[i][0, 0] - mid[i, 0]) > 0 and
                                                 abs(D[i][0, 0] - mid[i, 0]) < abs(O[i][0, 0] - mid[i, 0]) for i in both]))
    # fingers (dominant hand)
    gaps, spread, thumb, curl, back, palm_z = [], [], [], [], [], []
    for i in di:
        h = D[i]
        pw = np.linalg.norm(h[5, :2] - h[17, :2]) or 1e-3
        gaps.append(np.mean([np.linalg.norm(h[a, :2] - h[b, :2]) for a, b in ((8, 12), (12, 16), (16, 20))]) / pw)
        v1, v2 = h[8, :2] - h[5, :2], h[20, :2] - h[17, :2]
        spread.append(np.degrees(np.arccos(np.clip(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9), -1, 1))))
        thumb.append(np.linalg.norm(h[4, :2] - h[8, :2]) / pw)
        curl.append(np.mean([np.linalg.norm(h[t, :2] - h[0, :2]) for t in TIPS[1:]]) / (np.linalg.norm(h[9, :2] - h[0, :2]) or 1e-3))
        back.append(_angles_back(h, dom_right) > 0)
        n = np.cross(h[5] - h[0], h[17] - h[0])
        palm_z.append(n[2] / (np.linalg.norm(n) + 1e-9))
    # each finger's extension: tip-to-wrist over knuckle-to-wrist (straight ~1.8-2.2, curled toward 1)
    for name, (mcp, tip) in {"t": (2, 4), "i": (5, 8), "m": (9, 12), "r": (13, 16), "p": (17, 20)}.items():
        out[f"ext_{name}"] = float(np.median([np.linalg.norm(D[i][tip, :2] - D[i][0, :2]) / (np.linalg.norm(D[i][mcp, :2] - D[i][0, :2]) or 1e-3)
                                              for i in di]))
    out["other_raised_frac"] = float(np.mean([O[i] is not None and O[i][0, 1] < hips_y[i] - 0.05 for i in span]))
    out.update({"finger_gap_mean": float(np.mean(gaps)), "finger_gap_min": float(np.min(gaps)),
                "finger_spread_deg": float(np.median(spread)), "thumb_index": float(np.median(thumb)),
                "finger_curl_med": float(np.median(curl)), "finger_curl_range": float(np.ptp(curl)),
                "finger_back_frac": float(np.mean(back)), "palm_facing_cam": float(np.median(palm_z)),
                "palm_turn": float(np.ptp(palm_z))})
    # movement
    sp = np.array([np.linalg.norm(D[b][0, :2] - D[a][0, :2]) / sw / max(frames[b].t - frames[a].t, 1) * 1000
                   for a, b in zip(di, di[1:])])
    out.update({"path_sw": float(np.sum(sp * np.diff([frames[i].t for i in di]) / 1000)) if len(sp) else 0.0,
                "speed_mean": float(sp.mean()) if len(sp) else 0.0, "speed_peak": float(np.percentile(sp, 95)) if len(sp) else 0.0,
                "hold_frac": float(np.mean(sp < 0.3)) if len(sp) else 1.0})
    if len(sp) > 4:
        k = np.convolve(sp, np.ones(3) / 3, mode="same")
        peaks = [j for j in range(1, len(k) - 1) if k[j] > k[j - 1] and k[j] >= k[j + 1] and k[j] > 0.5 * k.max()]
        out["repetitions"] = len(peaks)
    tips = np.array([D[i][list(TIPS), :2] for i in di])
    if len(tips) > 2:
        acc = np.linalg.norm(np.diff(tips, n=2, axis=0), axis=2).mean(1) / sw
        out["finger_jitter"] = float(np.median(acc))
    out["face_width_px"] = float(np.median([abs(f.pose[7][0] - f.pose[8][0]) for f in frames]) * (meta.get("w") or 0))  # ear to ear
    return out
