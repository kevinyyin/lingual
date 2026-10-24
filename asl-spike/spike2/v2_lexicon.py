"""V2/V3: build avatar/signs2.json for avatar/index2.html (local only: derived from ASL Citizen).

Per class (the V1 lexicon clip): all frames (pose world, pose 2D with visibility, both hands with x scaled to
pixel proportions), the V1 kept-frame indices (trim + hold compression), the data's dominant side,
ASL-LEX sign type, dominant and non-dominant handshape (+ data-derived end shape for flexion-change signs).
Global: the handshape library (median joint angles), non-dominant label aliases, and headshake parameters
measured on round-1 S3 negations (yaw-proxy amplitude and dominant frequency in 1.5-6 Hz).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE2, R2, ROOT  # noqa: E402
import trim as TR  # noqa: E402

ND_ALIAS = {"B": "open_b", "C": "c", "8": "open_8", "stacked_5": "5", "DominanceConditionViolation": None}


def r4(a):
    return None if np.isnan(a).all() else np.round(np.nan_to_num(a), 4).tolist()


def headshake_params():
    """dominant headshake frequency and yaw amplitude on round-1 RWTH negations."""
    sys.path.insert(0, str(ROOT / "spike"))
    from s3_face import signals, signing_window
    res = json.load(open(ROOT / "results/s3_rwth_eval.json"))["primary_exact"]["per_utterance"]
    freqs, amps = [], []
    for u in res:
        if not u["neg"]:
            continue
        d = dict(np.load(ROOT / "cache/s3/rwth" / f"{u['name']}.npz", allow_pickle=True))
        fps = float(d["fps"]) or 25.0
        sig = signals(d)
        a, b = signing_window(sig, fps)
        y = sig["yaw"][a:b]
        y = y[np.isfinite(y)]
        if len(y) < 16:
            continue
        k = max(3, int(fps * 0.8))
        hp = y - np.convolve(np.pad(y, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")
        f = np.fft.rfftfreq(len(hp), 1 / fps)
        P = np.abs(np.fft.rfft(hp)) ** 2
        band = (f >= 1.0) & (f <= 6)
        if band.any():
            freqs.append(float(f[band][np.argmax(P[band])]))
        amps.append(float(np.std(hp)))
    amp_proxy = float(np.median(amps))
    # yaw proxy = nose offset / cheek-to-cheek width ~= 0.5 * sin(yaw) for a head of roughly circular cross-section
    peak_deg = float(np.degrees(np.arcsin(min(1.0, 2 * amp_proxy * np.sqrt(2)))))
    return {"freq_hz": round(float(np.median(freqs)), 2), "n": len(freqs), "yaw_proxy_sd": round(amp_proxy, 4),
            "peak_deg_estimate": round(peak_deg, 1),
            "note": "peak = asin(2 * sqrt(2) * proxy sd), assuming proxy ~ 0.5 sin(yaw) and a sinusoidal shake"}


def main():
    links = json.load(open(R2 / "asllex_links.json"))["classes"]
    lexv1 = json.load(open(CACHE2 / "lexicon_v1.json"))
    trims = json.load(open(CACHE2 / "trim_citizen.json"))
    hs = json.load(open(R2 / "handshapes.json"))
    out = {}
    for c, e in lexv1.items():
        d = dict(np.load(ROOT / "cache/s2/citizen" / f"{os.path.splitext(e['video'])[0]}.npz"))
        asp = float(d["width"]) / float(d["height"])
        lh, rh = d["lh"].copy(), d["rh"].copy()
        for h in (lh, rh):
            h[..., 0] *= asp
            h[..., 2] *= asp
        p2 = d["pose"].copy()
        p2[..., 0] *= asp
        dom_left = TR._motion(d["lh"]) > TR._motion(d["rh"])
        prim = links[c]["primary"] or {}
        frames = [{"pw": r4(d["pose_world"][i]), "p2": r4(p2[i]), "lh": r4(lh[i]), "rh": r4(rh[i])} for i in range(len(lh))]
        nd = prim.get("nondominant_handshape")
        nd = ND_ALIAS.get(nd, nd)
        out[c] = {"fps": e["fps"], "frames": frames, "keep": e["frames"], "active": [trims[e["video"]]["a"], trims[e["video"]]["b"]],
                  "dom_side_data": "left" if dom_left else "right", "sign_type": prim.get("sign_type"),
                  "hs_dom": prim.get("handshape"), "hs_nd": nd, "flex_change": bool(prim.get("flexion_change")),
                  "static": e["static"], "signer": e["signer"]}
    from collections import Counter
    thumb = {}
    for c, v in links.items():
        e = v["primary"]
        if e and e.get("handshape"):
            thumb.setdefault(e["handshape"], Counter())[str(e.get("thumb_position"))] += 1
    meta = {"thumb": {k: c.most_common(1)[0][0] for k, c in thumb.items()},
            "handshapes": {k: v["median"] for k, v in hs["library"].items()},
            "end_shapes": {k: v["median"] for k, v in hs["flexion_change_end_shapes"].items()},
            "headshake": headshake_params()}
    json.dump({"signs": out, "meta": meta}, open(ROOT / "avatar/signs2.json", "w"))
    print("signs", len(out), "size MB", round((ROOT / "avatar/signs2.json").stat().st_size / 1e6, 1), "headshake", meta["headshake"])


if __name__ == "__main__":
    main()
