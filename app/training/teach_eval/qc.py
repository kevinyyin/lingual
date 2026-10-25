"""Automated checks of each clip (step 1a): is it a usable one-sign video?

From the video: size, frame rate, duration, blur (variance of the Laplacian), brightness, a caption band (edge density
of the lower third against the rest). From the landmarks: how often a body / any hand / both hands are found, the face's
width in pixels, and how many signs the app's own segmenter cuts it into (one is expected). Flags, not verdicts.
"""
import json

import numpy as np

from training.eval_live import run_turn
from training.landmarks import load_npz
from training.teach_eval.clips import Clip

FLAGS = {  # name: (feature, test)
    "few_hands": ("hand_frac", lambda v: v < 0.3),
    "small_in_frame": ("face_width_px", lambda v: v < 40),
    "not_one_sign": ("segments", lambda v: v != 1),
    "dark": ("brightness", lambda v: v < 50),
    "blurry": ("blur", lambda v: v < 15),
    "too_short": ("duration_s", lambda v: v < 0.4),
    "too_long": ("duration_s", lambda v: v > 8),
    "caption_band": ("caption_band", lambda v: v > 2.0),
}


def _pixels(path: str, n: int = 6) -> dict:
    import cv2
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    blur, bright, band = [], [], []
    for k in np.linspace(0, total - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
        ok, img = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur.append(cv2.Laplacian(g, cv2.CV_64F).var())
        bright.append(g.mean())
        e = cv2.Canny(g, 80, 160)
        h = g.shape[0]
        band.append((e[2 * h // 3:].mean() + 1e-3) / (e[: 2 * h // 3].mean() + 1e-3))
    cap.release()
    med = lambda xs: float(np.median(xs)) if xs else float("nan")  # noqa: E731
    return {"blur": med(blur), "brightness": med(bright), "caption_band": med(band)}


def check(c: Clip) -> dict:
    meta = json.loads(c.frames_path("native").with_suffix(".json").read_text())
    native, browser = load_npz(c.frames_path("native")), load_npz(c.frames_path("15fps"))
    n_video = max(meta.get("n") or len(native), 1)
    ev = run_turn(browser) if browser else None
    out = {"id": c.id, "source": c.source, "gloss": c.gloss, "fps": meta.get("fps"), "w": meta.get("w"), "h": meta.get("h"),
           "duration_s": (native[-1].t - native[0].t) / 1000 if len(native) > 1 else 0.0,
           "pose_frac": len(native) / n_video,
           "hand_frac": float(np.mean([f.left is not None or f.right is not None for f in native])) if native else 0.0,
           "both_frac": float(np.mean([f.left is not None and f.right is not None for f in native])) if native else 0.0,
           "face_width_px": float(np.median([abs(f.pose[7][0] - f.pose[8][0]) for f in native]) * (meta.get("w") or 0)) if native else 0.0,
           "segments": len(ev.segments) if ev else 0, **_pixels(c.path)}
    out["flags"] = [name for name, (k, bad) in FLAGS.items() if k in out and out[k] == out[k] and bad(out[k])]
    return out
