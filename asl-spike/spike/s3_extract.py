"""S3 extraction: per utterance video, Holistic landmarks (full 478-point face + 52 blendshapes, hands,
pose) and a frame-by-frame naive facial-expression baseline (HSEmotion enet_b0_8, AffectNet 8 classes) on
a face crop. Output: cache/s3/<source>/<name>.npz.

Sources:
  rwth   RWTH-BOSTON-104 camera0 (full-body view, 336x312 with a text overlay at the bottom). We crop the
         top 240 rows and upscale 2x before MediaPipe; faces are ~40 px wide in the original.
  h2s    How2Sign frontal sentence clips (1280x720), if downloaded.
"""
from __future__ import annotations

import os
import sys
import time
import urllib.request  # noqa: F401  (hsemotion_onnx needs it imported)
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from extract import holistic_video  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HSE_CLASSES = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]


def face_crops(path, crop, upscale, face):
    """Square face crops (224 px) around the face landmarks, per frame; None when no face."""
    from extract import read_frames
    frames, _ = read_frames(path, crop=crop, upscale=upscale)
    out = []
    for fr, f in zip(frames, face):
        if np.isnan(f).all():
            out.append(None)
            continue
        h, w = fr.shape[:2]
        xs, ys = f[:, 0] * w, f[:, 1] * h
        cx, cy = np.nanmean(xs), np.nanmean(ys)
        s = 1.3 * max(np.nanmax(xs) - np.nanmin(xs), np.nanmax(ys) - np.nanmin(ys))
        x0, y0 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
        x1, y1 = int(min(w, cx + s / 2)), int(min(h, cy + s / 2))
        if x1 - x0 < 8 or y1 - y0 < 8:
            out.append(None)
            continue
        out.append(cv2.resize(fr[y0:y1, x0:x1], (224, 224), interpolation=cv2.INTER_CUBIC))
    return out


def one(args):
    src, path, name, crop, upscale = args
    dst = ROOT / "cache" / "s3" / src / f"{name}.npz"
    if dst.exists():
        return name, "cached"
    try:
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
        t0 = time.perf_counter_ns()
        d = holistic_video(path, crop=crop, upscale=upscale, keep_full_face=True)
        rec = HSEmotionRecognizer(model_name="enet_b0_8_best_vgaf")
        crops = face_crops(path, crop, upscale, d["face"])
        hse = np.full((len(crops), 8), np.nan, np.float32)
        for i, c in enumerate(crops):
            if c is not None:
                hse[i] = rec.predict_emotions(c, logits=False)[1]
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst, pose=d["pose"], lh=d["lh"], rh=d["rh"], face=d["face"], blend=d["blend"],
                            blend_names=np.array(d["blend_names"]), hse=hse, fps=d["fps"], width=d["width"],
                            height=d["height"], extract_ms=d["extract_ms"],
                            total_ms=(time.perf_counter_ns() - t0) / 1e6)
        return name, "ok"
    except Exception as e:  # record, never drop
        return name, f"failed: {type(e).__name__}: {e}"


def rwth_jobs():
    vb = ROOT / "data/rwth-boston-104/rwth-boston-104/videoBank/camera0"
    return [("rwth", str(p), p.stem.split("_")[0], (0, 0, 336, 240), 2.0) for p in sorted(vb.glob("*.mpg"))]


def h2s_jobs(limit_ids=None):
    base = ROOT / "data/how2sign"
    vids = sorted(base.rglob("*.mp4"))
    if limit_ids is not None:
        vids = [v for v in vids if v.stem in limit_ids]
    return [("h2s", str(p), p.stem, None, 1.0) for p in vids]


def run(jobs, workers=6):
    import json
    log = []
    with Pool(workers) as pool:
        for i, (name, status) in enumerate(pool.imap_unordered(one, jobs)):
            log.append({"name": name, "status": status})
            if status.startswith("failed"):
                print(name, status, flush=True)
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{len(jobs)}", flush=True)
    return log


if __name__ == "__main__":
    import json
    which = sys.argv[1] if len(sys.argv) > 1 else "rwth"
    if which == "rwth":
        log = run(rwth_jobs())
    else:
        ids = None
        if len(sys.argv) > 2:
            ids = set(json.load(open(sys.argv[2])))
        log = run(h2s_jobs(ids))
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(log, open(ROOT / "results" / f"s3_extract_log_{which}.json", "w"), indent=0)
    print("done", sum(1 for x in log if x["status"] in ("ok", "cached")), "of", len(log))
