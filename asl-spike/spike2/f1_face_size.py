"""F1 (optional): how big must the face be for brow readings?

Clips: ASL Citizen test-split clips of wh-signs (WHAT, WHERE, WHO, WHY, HOW, WHICH, WHEN, HOW-MUCH, HOW-MANY) and
an equal, seeded sample of non-wh signs (up to 60 each). Every 2nd frame is processed. Each frame is rescaled so
the face bounding-box width (mesh points 234-454, from the cached landmarks) is 40, 60, 80, 120 or 160 px (160
may be an upscale), then run through the MediaPipe Face Landmarker with blendshapes; "native" = original frame.
Per scale: face-detection rate; Pearson correlation of browDown (mean L/R) and browInnerUp series with native;
AUC of sustained brow lowering (max of a 0.5 s rolling mean of browDown) for wh vs non-wh.
Distance: face_px = W * 0.15 m / (2 * d * tan(HFOV / 2)) for a 1080p camera (W = 1920 landscape, 1080 portrait).
"""
from __future__ import annotations

import json
import math
import os
import sys

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, citizen_class_map, citizen_rows, jdump  # noqa: E402

WH = {"WHAT", "WHERE", "WHO", "WHY", "HOW", "WHICH", "WHEN", "HOW-MUCH", "HOW-MANY"}
SCALES = ["native", 40, 60, 80, 120, 160]


def run_clip(video, face_w_native, fps):
    sys.path.insert(0, str(ROOT / "spike"))
    from extract import _make
    cap = cv2.VideoCapture(str(ROOT / "data/asl-citizen/videos_vocab" / video))
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % 2 == 0:
            frames.append(f)
        i += 1
    out = {}
    for sc in SCALES:
        mp, lm = _make("face")
        k = 1.0 if sc == "native" else sc / face_w_native
        dn, di, det = [], [], 0
        with lm:
            for j, f in enumerate(frames):
                g = f if k == 1.0 else cv2.resize(f, None, fx=k, fy=k, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
                r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(g, cv2.COLOR_BGR2RGB))), int(j * 2000 / fps))
                if r.face_blendshapes:
                    det += 1
                    b = {c.category_name: c.score for c in r.face_blendshapes[0]}
                    dn.append((b["browDownLeft"] + b["browDownRight"]) / 2)
                    di.append(b["browInnerUp"])
                else:
                    dn.append(np.nan)
                    di.append(np.nan)
        out[str(sc)] = {"det": det / max(1, len(frames)), "down": dn, "inner": di}
    return out


def sustained(x, fps):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    w = max(1, int(fps / 2 * 0.5))  # every 2nd frame
    if len(x) == 0:
        return np.nan
    return float(np.max(np.convolve(x, np.ones(w) / w, mode="valid"))) if len(x) >= w else float(np.mean(x))


def main(n=60):
    cmap = citizen_class_map()
    rows = citizen_rows()
    rows = rows[(rows["split"] == "test") & rows["Gloss"].isin(cmap)].copy()
    rows["cls"] = rows["Gloss"].map(cmap)
    wh = rows[rows["cls"].isin(WH)].sample(frac=1, random_state=0).head(n)
    non = rows[~rows["cls"].isin(WH)].sample(len(wh), random_state=0)
    clips = [(r.video, 1) for r in wh.itertuples()] + [(r.video, 0) for r in non.itertuples()]
    res = []
    for i, (v, lab) in enumerate(clips):
        d = np.load(ROOT / "cache/s2/citizen" / f"{os.path.splitext(v)[0]}.npz")
        fw = float(np.nanmedian(np.abs(d["face"][:, 23, 0] - d["face"][:, 22, 0]))) * float(d["width"])
        fps = float(d["fps"]) or 30.0
        if not np.isfinite(fw) or fw < 10:
            continue
        r = run_clip(v, fw, fps)
        res.append({"video": v, "wh": lab, "face_w_native": fw, "fps": fps, "r": r})
        if (i + 1) % 20 == 0:
            print(i + 1, "of", len(clips), flush=True)
    summ = {}
    for sc in SCALES:
        s = str(sc)
        det = [x["r"][s]["det"] for x in res]
        cors_d, cors_i = [], []
        for x in res:
            a, b = np.array(x["r"]["native"]["down"]), np.array(x["r"][s]["down"])
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() > 5 and np.std(a[m]) > 1e-6 and np.std(b[m]) > 1e-6:
                cors_d.append(np.corrcoef(a[m], b[m])[0, 1])
            a, b = np.array(x["r"]["native"]["inner"]), np.array(x["r"][s]["inner"])
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() > 5 and np.std(a[m]) > 1e-6 and np.std(b[m]) > 1e-6:
                cors_i.append(np.corrcoef(a[m], b[m])[0, 1])
        feat = [(sustained(x["r"][s]["down"], x["fps"]), x["wh"]) for x in res]
        feat = [(f, l) for f, l in feat if np.isfinite(f)]
        auc = roc_auc_score([l for _, l in feat], [f for f, _ in feat]) if len({l for _, l in feat}) == 2 else None
        summ[s] = {"detection_rate_mean": round(float(np.mean(det)), 3), "corr_browDown_median": round(float(np.median(cors_d)), 3) if cors_d else None,
                   "corr_browInnerUp_median": round(float(np.median(cors_i)), 3) if cors_i else None,
                   "auc_wh_sustained_browDown": round(float(auc), 3) if auc is not None else None, "n_clips_scored": len(feat)}
    nat_auc = summ["native"]["auc_wh_sustained_browDown"] or 0
    ok = [sc for sc in SCALES[1:] if summ[str(sc)]["detection_rate_mean"] >= 0.9 and (summ[str(sc)]["corr_browDown_median"] or 0) >= 0.8
          and (summ[str(sc)]["auc_wh_sustained_browDown"] or 0) >= nat_auc - 0.05]
    min_w = min(ok) if ok else None
    dist = {}
    if min_w:
        for orient, W in (("landscape_1920", 1920), ("portrait_1080", 1080)):
            for hfov in (70, 80):
                dist[f"{orient}_hfov{hfov}"] = round(W * 0.15 / (2 * min_w * math.tan(math.radians(hfov / 2))), 2)
    out = {"n_wh": int(sum(x["wh"] for x in res)), "n_non_wh": int(sum(1 - x["wh"] for x in res)),
           "signers": len({v.split("-")[0] for v, _ in clips}),
           "native_face_width_px_median": round(float(np.median([x["face_w_native"] for x in res])), 1),
           "by_scale": summ, "min_reliable_face_width_px": min_w,
           "criterion": "detection >= 90%, browDown correlation with native >= 0.8, wh AUC within 0.05 of native",
           "max_distance_m": dist, "assumptions": "face width 0.15 m; 1080p camera; HFOV 70-80 deg (landscape) - portrait uses the 1080-px width"}
    jdump(out, "f1_face_size.json")
    print(json.dumps({k: out[k] for k in ("native_face_width_px_median", "by_scale", "min_reliable_face_width_px", "max_distance_m")}, indent=1))


if __name__ == "__main__":
    main()
