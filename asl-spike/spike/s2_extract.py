"""S2 extraction: MediaPipe Holistic on the ASL Citizen videos of the vocabulary glosses (all splits).
Videos are unpacked from the zip on demand into data/asl-citizen/videos_vocab/ (never redistributed).
Per video we cache landmarks (hands, pose, pose_world, 24-point face subset, blendshapes) to
cache/s2/citizen/<video>.npz. Parallel workers on CPU."""
from __future__ import annotations

import json
import os
import sys
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from extract import holistic_video  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ZIP = ROOT / "data/asl-citizen/ASL_Citizen.zip"
VID = ROOT / "data/asl-citizen/videos_vocab"
OUT = ROOT / "cache/s2/citizen"


def one(name: str):
    dst = OUT / f"{Path(name).stem}.npz"
    if dst.exists():
        try:
            with np.load(dst) as z:
                _ = z["rh"].shape
            return name, "cached", 0.0
        except Exception:
            dst.unlink()  # partial file from an interrupted run
    try:
        d = holistic_video(str(VID / name))
        tmp = dst.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **{k: d[k] for k in ("pose", "pose_world", "lh", "rh", "face", "blend")},
                            fps=d["fps"], n_frames=d["n_frames"], width=d["width"], height=d["height"],
                            extract_ms=d["extract_ms"])
        os.replace(tmp, dst)
        return name, "ok", d["extract_ms"]
    except Exception as e:
        return name, f"failed: {type(e).__name__}: {e}"[:300], 0.0


def main(workers: int = 6):
    vocab = json.load(open(ROOT / "vocab.json"))
    want = {g for e in vocab for g in e["asl_citizen_class_videos"]}
    sp = ROOT / "data/asl-citizen/ASL_Citizen/splits"
    rows = pd.concat([pd.read_csv(sp / f"{s}.csv").assign(split=s) for s in ("train", "val", "test")])
    rows = rows[rows["Gloss"].isin(want)]
    rows.to_csv(ROOT / "results/s2_citizen_subset.csv", index=False)
    print("videos:", len(rows), "glosses:", rows["Gloss"].nunique(), "signers:", rows["Participant ID"].nunique(), flush=True)
    VID.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(ZIP)
    members = {Path(n).name: n for n in z.namelist() if n.endswith(".mp4")}
    names = []
    for v in rows["Video file"]:
        if not (VID / v).exists():
            with z.open(members[v]) as src, open(VID / v, "wb") as dst:
                dst.write(src.read())
        names.append(v)
    print("unpacked", len(names), flush=True)
    log, t0 = [], time.time()
    with Pool(workers) as pool:
        for i, (n, st, ms) in enumerate(pool.imap_unordered(one, names, chunksize=4)):
            log.append({"video": n, "status": st, "extract_ms": ms})
            if st.startswith("failed"):
                print(n, st, flush=True)
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(names)} {time.time() - t0:.0f}s", flush=True)
    json.dump(log, open(ROOT / "results/s2_extract_log.json", "w"))
    print("done", sum(1 for x in log if x["status"] in ("ok", "cached")), "of", len(log))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
