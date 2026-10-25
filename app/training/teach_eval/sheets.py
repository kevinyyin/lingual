"""Contact sheets for the reviewing agents: key frames of a clip, and of a reference video of the same (or the
predicted) sign. JPEGs in app/data/teaching/sheets (local only; dataset frames are never committed or uploaded)."""
import csv
import re
from pathlib import Path

import numpy as np

from server.vocab import load_vocab
from training.teach_eval.clips import DATA, ROOT

SHEETS = DATA / "sheets"
CITIZEN_TRAIN = ROOT / "data" / "asl_citizen" / "splits" / "train.csv"
CITIZEN_VIDEOS = [ROOT / "data" / "asl_citizen" / "videos", ROOT.parents[1] / "lingual" / "asl-spike" / "data" / "asl-citizen" / "videos_vocab"]
CITIZEN_ZIP = ROOT.parents[1] / "lingual" / "asl-spike" / "data" / "asl-citizen" / "ASL_Citizen.zip"  # the full dataset
N, H = 6, 300  # frames per sheet, frame height in pixels


def signer_box(frames, t0=None, t1=None, pad=0.12):
    """The signer's box in normalised image coordinates over the active span: pose (head to hips) and hands, padded."""
    pts = []
    for f in frames:
        if (t0 is None or f.t >= t0 - 300) and (t1 is None or f.t <= t1 + 300):
            pts += [p[:2] for p in f.pose[:25]] + [p[:2] for h in (f.left, f.right) if h for p in h]
    if not pts:
        return None
    a = np.array(pts)
    lo, hi = np.percentile(a, 1, 0), np.percentile(a, 99, 0)
    d = hi - lo
    return (max(0.0, lo[0] - pad * d[0] - 0.04), max(0.0, lo[1] - pad * d[1] - 0.04), min(1.0, hi[0] + pad * d[0] + 0.04), min(1.0, hi[1] + pad * d[1]))


def _frames(path: str, t0: float | None, t1: float | None, n: int = N, box=None) -> list:
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    lo = int((t0 or 0) / 1000 * fps) if t0 is not None else 0
    hi = int((t1 or 0) / 1000 * fps) if t1 is not None else total - 1
    pad = max(2, (hi - lo) // 8)  # a little before and after the active span
    out = []
    for k in np.linspace(max(0, lo - pad), min(total - 1, hi + pad), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
        ok, img = cap.read()
        if ok:
            if box:  # crop to the signer: on a wide video the hands are otherwise a few pixels
                hh, ww = img.shape[:2]
                img = img[int(box[1] * hh):max(int(box[3] * hh), int(box[1] * hh) + 2), int(box[0] * ww):max(int(box[2] * ww), int(box[0] * ww) + 2)]
            out.append((k / fps, cv2.resize(img, (int(img.shape[1] * H / img.shape[0]), H))))
    cap.release()
    return out


def _strip(frames: list, title: str):
    import cv2
    cells = []
    for t, img in frames:
        img = img.copy()
        cv2.putText(img, f"{t:.2f}s", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cells.append(img)
    strip = np.hstack(cells) if cells else np.zeros((H, 400, 3), np.uint8)
    bar = np.full((34, strip.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return np.vstack([bar, strip])


def reference_video(gloss: str) -> str | None:
    """A training video of the sign from ASL Citizen (a different signer than any test clip)."""
    sign = load_vocab(extended=None).get(gloss)
    want = re.sub(r"[^A-Z]", "", (sign.asl_citizen if sign else gloss).upper())
    with open(CITIZEN_TRAIN, newline="", encoding="utf-8") as fh:
        for row in list(csv.reader(fh))[1:]:
            if re.sub(r"[^A-Z]", "", re.sub(r"\d+$", "", row[2].upper())) == want:
                for d in CITIZEN_VIDEOS:
                    if (d / row[1]).exists():
                        return str(d / row[1])
                if CITIZEN_ZIP.exists():  # not unpacked yet: take this one video out of the local dataset archive
                    import zipfile
                    dst = CITIZEN_VIDEOS[0] / row[1]
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(CITIZEN_ZIP) as z:
                        dst.write_bytes(z.read(f"ASL_Citizen/videos/{row[1]}"))
                    return str(dst)
    return None


def clip_sheet(clip_path: str, title: str, t0: float | None, t1: float | None, out: Path, box=None):
    import cv2
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), _strip(_frames(clip_path, t0, t1, box=box), title), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return out


def pair_sheet(clip_path: str, t0, t1, ref_gloss: str, top_title: str, ref_title: str, out: Path, box=None) -> Path:
    """The clip's frames (cropped to `box`, the signer) above a reference video's frames for `ref_gloss` (the label,
    or the predicted sign)."""
    import cv2
    top = _strip(_frames(clip_path, t0, t1, box=box), top_title)
    ref = reference_video(ref_gloss)
    bottom = _strip(_frames(ref, None, None), ref_title) if ref else _strip([], f"{ref_title}: no reference video")
    w = max(top.shape[1], bottom.shape[1])
    pad = lambda a: np.pad(a, ((0, 0), (0, w - a.shape[1]), (0, 0)))  # noqa: E731
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack([pad(top), np.full((8, w, 3), 255, np.uint8), pad(bottom)]), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return out
