"""S3 per-utterance face/kinematic signals and the two layers.

Signals (per frame, from MediaPipe Holistic face mesh + blendshapes):
  brow_up    mean(browInnerUp, browOuterUpLeft, browOuterUpRight)
  brow_down  mean(browDownLeft, browDownRight)
  brow_h     geometric brow height: mean(brow point -> upper eyelid distance) / inter-ocular distance
  yaw, pitch, roll   head pose proxies from the mesh (nose vs cheek outline; eye line angle)
Affect-only signals (deliberately excluding brows, which are grammar in ASL):
  smile, frown, cheek_squint, eye_squint, nose_sneer, mouth_press, jaw_open
Kinematics (hands): speed, amplitude, tempo (speed peaks per second).

Grammar layer: sentence type from sustained brow raise vs lower over the signing window (not single
frames), with head pitch as a secondary cue; negation from headshake (yaw oscillation).
Signals are z-scored per signer over all of that signer's frames (unsupervised calibration; no labels).
"""
from __future__ import annotations

import numpy as np

BROW_L = [70, 63, 105, 66, 107]
BROW_R = [300, 293, 334, 296, 336]
LID_L, LID_R = 159, 386
EYE_OUT_L, EYE_OUT_R = 33, 263
NOSE, CHEEK_L, CHEEK_R, FOREHEAD, CHIN = 1, 234, 454, 10, 152

AFFECT_BS = {"smile": ["mouthSmileLeft", "mouthSmileRight"], "frown": ["mouthFrownLeft", "mouthFrownRight"],
             "cheek_squint": ["cheekSquintLeft", "cheekSquintRight"], "eye_squint": ["eyeSquintLeft", "eyeSquintRight"],
             "nose_sneer": ["noseSneerLeft", "noseSneerRight"], "mouth_press": ["mouthPressLeft", "mouthPressRight"],
             "jaw_open": ["jawOpen"]}


def _bs(blend, names, keys):
    idx = [names.index(k) for k in keys if k in names]
    return np.nanmean(blend[:, idx], 1) if idx else np.full(len(blend), np.nan)


def signals(d: dict) -> dict[str, np.ndarray]:
    names = [str(n) for n in d["blend_names"]]
    blend, face = d["blend"], d["face"]
    out = {"brow_up": _bs(blend, names, ["browInnerUp", "browOuterUpLeft", "browOuterUpRight"]),
           "brow_down": _bs(blend, names, ["browDownLeft", "browDownRight"])}
    io = np.linalg.norm(face[:, EYE_OUT_L, :2] - face[:, EYE_OUT_R, :2], axis=-1)
    hl = np.linalg.norm(face[:, BROW_L, :2] - face[:, [LID_L], :2], axis=-1).mean(1)
    hr = np.linalg.norm(face[:, BROW_R, :2] - face[:, [LID_R], :2], axis=-1).mean(1)
    out["brow_h"] = (hl + hr) / 2 / io
    cw = face[:, CHEEK_R, 0] - face[:, CHEEK_L, 0]
    out["yaw"] = (face[:, NOSE, 0] - (face[:, CHEEK_L, 0] + face[:, CHEEK_R, 0]) / 2) / cw
    fh = face[:, CHIN, 1] - face[:, FOREHEAD, 1]
    eye_mid_y = (face[:, EYE_OUT_L, 1] + face[:, EYE_OUT_R, 1]) / 2
    out["pitch"] = (face[:, NOSE, 1] - eye_mid_y) / fh
    out["roll"] = np.arctan2(face[:, EYE_OUT_R, 1] - face[:, EYE_OUT_L, 1], face[:, EYE_OUT_R, 0] - face[:, EYE_OUT_L, 0])
    for k, v in AFFECT_BS.items():
        out[k] = _bs(blend, names, v)
    # hand kinematics (wrists from the hand landmarks, normalized by inter-ocular distance in px space)
    wr = np.stack([d["lh"][:, 0, :2], d["rh"][:, 0, :2]], 1)
    sp = np.linalg.norm(np.diff(wr, axis=0), axis=-1) / np.nanmedian(io)
    out["hand_speed"] = np.concatenate([[np.nan], np.nanmax(np.where(np.isnan(sp), -1, sp), 1)])
    out["hand_speed"][out["hand_speed"] < 0] = np.nan
    out["hands_present"] = (~np.isnan(d["lh"][:, 0, 0]) | ~np.isnan(d["rh"][:, 0, 0])).astype(float)
    out["hand_y"] = np.nanmin(np.where(np.isnan(wr[..., 1]), np.inf, wr[..., 1]), 1)
    out["hand_y"][~np.isfinite(out["hand_y"])] = np.nan
    return out


def signing_window(sig: dict, fps: float) -> tuple[int, int]:
    """Frames between the first and last frame with a visible hand that moves (rest frames trimmed)."""
    hp = sig["hands_present"] > 0
    sp = np.nan_to_num(sig["hand_speed"], nan=0.0)
    active = hp & (sp > np.nanpercentile(sp[hp], 20) if hp.any() else hp)
    idx = np.where(active)[0]
    if len(idx) < 3:
        return 0, len(hp)
    return int(idx[0]), int(idx[-1]) + 1


def headshake(yaw: np.ndarray, fps: float) -> dict:
    y = yaw.copy()
    ok = ~np.isnan(y)
    if ok.sum() < 8:
        return {"amp": 0.0, "reversals_per_s": 0.0, "band_ratio": 0.0}
    y[~ok] = np.interp(np.where(~ok)[0], np.where(ok)[0], y[ok])
    k = max(3, int(fps * 0.8))
    trend = np.convolve(np.pad(y, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")
    hp = y - trend
    amp = float(np.std(hp))
    s = np.sign(hp)
    big = np.abs(hp) > 0.5 * amp
    rev = int(np.sum((s[1:] != s[:-1]) & big[1:]))
    f = np.fft.rfftfreq(len(hp), 1 / fps)
    P = np.abs(np.fft.rfft(hp)) ** 2
    band = P[(f >= 1.5) & (f <= 6)].sum() / max(P[f > 0.3].sum(), 1e-9)
    return {"amp": amp, "reversals_per_s": rev / (len(hp) / fps), "band_ratio": float(band)}


def utterance_features(sig: dict, z: dict, fps: float) -> dict:
    """z: per-signer (mean, std) for each signal, used to standardize."""
    a, b = signing_window(sig, fps)
    n = b - a
    feats = {"frames": n, "seconds": n / fps}

    def zs(k):
        m, s = z[k]
        return (sig[k][a:b] - m) / (s if s > 1e-9 else 1.0)

    for k in ("brow_up", "brow_down", "brow_h", "pitch"):
        v = zs(k)
        v = v[~np.isnan(v)]
        if len(v) == 0:
            v = np.zeros(1)
        # sustained level: 0.5 s rolling mean, then its max/min; plus overall mean and the final 40%
        w = max(1, int(fps * 0.5))
        roll = np.convolve(v, np.ones(w) / w, mode="valid") if len(v) >= w else v
        feats[f"{k}_mean"] = float(np.mean(v))
        feats[f"{k}_sus_max"] = float(np.max(roll))
        feats[f"{k}_sus_min"] = float(np.min(roll))
        feats[f"{k}_end"] = float(np.mean(v[int(0.6 * len(v)):]))
    hs = headshake(sig["yaw"][a:b], fps)
    feats.update({f"hs_{k}": v for k, v in hs.items()})
    for k in AFFECT_BS:
        v = zs(k)
        feats[f"aff_{k}"] = float(np.nanmean(v)) if np.isfinite(v).any() else 0.0
    sp = sig["hand_speed"][a:b]
    feats["kin_speed"] = float(np.nanmedian(sp)) if np.isfinite(sp).any() else 0.0
    feats["kin_amp"] = float(np.nanpercentile(sp, 90)) if np.isfinite(sp).any() else 0.0
    # tempo: speed peaks per second (one peak ~ one sign stroke)
    s2 = np.nan_to_num(sp)
    pk = np.sum((s2[1:-1] > s2[:-2]) & (s2[1:-1] >= s2[2:]) & (s2[1:-1] > np.median(s2))) if len(s2) > 2 else 0
    feats["kin_tempo"] = float(pk / max(feats["seconds"], 1e-3))
    return feats


def signer_norms(sigs: list[dict]) -> dict:
    keys = [k for k in sigs[0] if k not in ("hands_present",)]
    out = {}
    for k in keys:
        v = np.concatenate([s[k] for s in sigs])
        v = v[np.isfinite(v)]
        out[k] = (float(np.mean(v)) if len(v) else 0.0, float(np.std(v)) if len(v) else 1.0)
    return out


def affect_layer(f: dict, kin_ref: dict) -> dict:
    """Small rule set over non-brow face signals and kinematics -> {label, intensity}.
    kin_ref: per-signer median kinematics for ratios."""
    smile, frown = f["aff_smile"], f["aff_frown"]
    tension = 0.5 * (f["aff_nose_sneer"] + f["aff_mouth_press"])
    sp = f["kin_speed"] / max(kin_ref["kin_speed"], 1e-6)
    amp = f["kin_amp"] / max(kin_ref["kin_amp"], 1e-6)
    scores = {"happy": smile + 0.3 * f["aff_cheek_squint"],
              "sad": frown + 0.8 * max(0.0, 1 - sp),
              "angry": tension + 0.8 * max(0.0, amp - 1) + 0.3 * f["aff_eye_squint"]}
    lab, s = max(scores.items(), key=lambda kv: kv[1])
    if s < 0.6:
        lab = "neutral"
    inten = float(1 / (1 + np.exp(-(s - 0.6) * 2.5))) if lab != "neutral" else float(min(0.3, max(0.0, s / 2)))
    return {"label": lab, "intensity": round(inten, 3), "scores": {k: round(v, 3) for k, v in scores.items()},
            "tempo_ratio": round(float(sp), 3), "amplitude_ratio": round(float(amp), 3)}
