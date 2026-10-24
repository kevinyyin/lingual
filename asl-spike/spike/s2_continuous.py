"""S2 continuous-signing test on RWTH-BOSTON-104 (fallback for ASLLRP; no DAI access).

Ground truth: each sentence's gloss sequence (RWTH annotation, which comes from NCSLGR). The public RWTH
annotation has no per-sign times, so boundary error cannot be measured; we report recall/precision of
vocabulary signs by order-preserving alignment, false positives per out-of-vocabulary sign, and the delay
from the end of signing (hands to rest) to the recognized sequence.

Recognizer: sliding windows (lengths 12-36 frames at 30 fps, stride 3) over Holistic landmarks, each
classified by the ASL-Citizen model; windows below a confidence threshold are dropped; temporal
non-maximum suppression keeps the most confident window per region; consecutive duplicates merge.
Uses only sentences with at least two vocabulary signs.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from extract import FACE_SUBSET  # noqa: E402
from features import FACE_SUBSET_POS, POSE_IDX, normalize  # noqa: E402
from s2_model import SignNet  # noqa: E402
from s3_face import signals, signing_window  # noqa: E402
from vocab_src import V  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FACE_FULL_IDX = [FACE_SUBSET[i] for i in FACE_SUBSET_POS]


def norm(s):
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def alias_map(classes):
    m = {}
    for g in classes:
        for a in [g] + V[g][1]:
            m.setdefault(norm(a), g)
    return m


def lcs(a, b):
    D = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            D[i + 1][j + 1] = D[i][j] + 1 if a[i] == b[j] else max(D[i][j + 1], D[i + 1][j])
    return D[-1][-1]


def windows(d, fps):
    lh, rh = d["lh"][..., :2], d["rh"][..., :2]
    pose = d["pose"][:, POSE_IDX, :2]
    face = d["face"][:, FACE_FULL_IDX, :2]
    F = len(lh)
    scale = 30.0 / fps
    out = []
    for L in (12, 18, 24, 30, 36):
        Lf = max(6, int(round(L / scale)))
        for s in range(0, max(1, F - Lf + 1), max(1, int(round(3 / scale)))):
            sl = slice(s, s + Lf)
            out.append((s, s + Lf, normalize(lh[sl], rh[sl], pose[sl], face[sl])))
    return out


def recognize(model, classes, d, fps, thr):
    t0 = time.perf_counter_ns()
    ws = windows(d, fps)
    X = torch.tensor(np.stack([w[2] for w in ws]))
    with torch.no_grad():
        p = torch.softmax(torch.cat([model(X[i:i + 512]) for i in range(0, len(X), 512)]), 1).numpy()
    conf, lab = p.max(1), p.argmax(1)
    cand = sorted([(conf[i], ws[i][0], ws[i][1], classes[lab[i]]) for i in range(len(ws)) if conf[i] >= thr], reverse=True)
    keep = []
    for c, a, b, g in cand:  # NMS: drop windows overlapping a kept one by > 30% of the shorter
        if all(min(b, kb) - max(a, ka) <= 0.3 * min(b - a, kb - ka) for _, ka, kb, _ in keep):
            keep.append((c, a, b, g))
    keep.sort(key=lambda k: k[1])
    seq = []
    for c, a, b, g in keep:
        if not seq or seq[-1] != g:
            seq.append(g)
    return seq, keep, (time.perf_counter_ns() - t0) / 1e6, len(ws)


def rest_detection_delay(d, fps, hold_ms=300):
    """Frames from the end of signing to when 'hands at rest' is declared (speed below the 20th percentile
    for hold_ms, or hands out of view)."""
    sig = signals({**d, "blend_names": d["blend_names"]})
    a, b = signing_window(sig, fps)
    return hold_ms, b


def main():
    ck = torch.load(ROOT / "cache/s2/model.pt")
    classes = ck["classes"]
    model = SignNet(len(classes))
    model.load_state_dict(ck["state"])
    model.eval()
    amap = alias_map(classes)
    rows = json.load(open(ROOT / "results/s3_rwth_ncslgr_match.json"))["rows"]
    res = {"thresholds": {}, "note": "boundary error not measurable: RWTH annotations have no per-sign times"}
    per_utt = {}
    data = {}
    for r in rows:
        toks = r["orth"].split()
        gt = [amap[norm(t)] for t in toks if norm(t) in amap]
        if len(gt) < 2:
            continue
        p = ROOT / "cache/s3/rwth" / f"{r['name']}.npz"
        if not p.exists():
            continue
        d = dict(np.load(p, allow_pickle=True))
        data[r["name"]] = (r, toks, gt, d, float(d["fps"]) or 30.0)
    for thr in (0.3, 0.5, 0.7, 0.85):
        tp = npred = ngt = noov = 0
        delays, comp = [], []
        for name, (r, toks, gt, d, fps) in data.items():
            seq, keep, ms, nw = recognize(model, classes, d, fps, thr)
            m = lcs(gt, seq)
            tp += m
            npred += len(seq)
            ngt += len(gt)
            noov += len(toks) - len(gt)
            comp.append(ms)
            per_utt.setdefault(name, {"gt": gt, "orth": r["orth"], "signer": r["speaker"]})[f"pred@{thr}"] = seq
        fp = npred - tp
        res["thresholds"][str(thr)] = {"recall": round(tp / ngt, 3), "precision": round(tp / max(npred, 1), 3),
                                        "tp": tp, "n_gt_vocab_tokens": ngt, "n_pred": npred,
                                        "false_positives": fp, "oov_tokens": noov,
                                        "fp_per_oov_token": round(fp / max(noov, 1), 3),
                                        "compute_ms_median": float(np.median(comp)), "compute_ms_p90": float(np.percentile(comp, 90))}
        print(thr, res["thresholds"][str(thr)], flush=True)
    res["n_utterances"] = len(data)
    res["n_signers"] = len({v[0]["speaker"] for v in data.values()})
    res["fps"] = sorted({v[4] for v in data.values()})
    res["delay_model"] = ("delay after the last sign = rest detection hold (300 ms of low hand speed) + windowed "
                          "classification compute of the whole utterance (compute_ms above, CPU)")
    res["per_utterance"] = per_utt
    json.dump(res, open(ROOT / "results/s2_continuous_rwth.json", "w"), indent=1)


if __name__ == "__main__":
    main()
