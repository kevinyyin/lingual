"""R1 follow-up: aspect-ratio test. MediaPipe's normalized coordinates divide x by image width and y by image
height, so the same pose gives different normalized geometry in landscape (ASL Citizen webcams) and portrait
(PopSign phones) video. Round 1 did not correct for this.

1. ASL Citizen aspect (w/h) is known per clip (cached width/height).
2. Estimate the PopSign aspect from the face mesh: face bbox (234-454 wide, 10-152 tall) has a fixed true
   width/height ratio, so a_pop = ratio_px / ratio_norm_pop with ratio_px = ratio_norm_cit * a_cit.
3. Re-extract the Kaggle features with x scaled by a_pop / a_cit_median (equivalent to giving PopSign the same
   pixel geometry as the training data) and re-score the unchanged round-1 model.
"""
from __future__ import annotations

import io
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, acc_with_ci, citizen_rows, jdump, load_model, vocab  # noqa: E402
from features import from_kaggle, normalize  # noqa: E402


def face_ratio_kaggle(df):
    f = df[(df.type == "face") & (df.landmark_index.isin([234, 454, 10, 152]))].pivot_table(index="frame", columns="landmark_index", values=["x", "y"])
    w = (f[("x", 454)] - f[("x", 234)]).abs()
    h = (f[("y", 152)] - f[("y", 10)]).abs()
    return float(np.nanmedian(w / h))


def main():
    rows = citizen_rows()
    # ASL Citizen: aspect and normalized face ratio (FACE_SUBSET positions: 234->22, 454->23, 10->21, 152->20)
    a_cit, r_cit = [], []
    for v in rows.sample(400, random_state=0)["video"]:
        p = ROOT / "cache/s2/citizen" / f"{os.path.splitext(v)[0]}.npz"
        if not p.exists():
            continue
        with np.load(p) as d:
            a_cit.append(float(d["width"]) / float(d["height"]))
            f = d["face"]
            with np.errstate(all="ignore"):
                r_cit.append(float(np.nanmedian(np.abs(f[:, 23, 0] - f[:, 22, 0]) / np.abs(f[:, 20, 1] - f[:, 21, 1]))))
    a_c = float(np.median(a_cit))
    ratio_px = float(np.nanmedian(np.array(r_cit) * np.array(a_cit)))
    classes = load_model()[1]
    v = {e["gloss"]: e for e in vocab()}
    k2v = {}
    for g in classes:
        for k in v[g]["kaggle_asl_signs"]:
            k2v.setdefault(k, g)
    meta = pd.read_csv(ROOT / "data/kaggle-asl-signs/train.csv")
    meta = meta[meta["sign"].isin(k2v)].reset_index(drop=True)
    z = zipfile.ZipFile(ROOT / "data/kaggle-asl-signs/asl-signs.zip")
    samp = meta.sample(600, random_state=1)
    r_pop = [face_ratio_kaggle(pd.read_parquet(io.BytesIO(z.read(p)))) for p in samp["path"]]
    a_p = ratio_px / float(np.nanmedian(r_pop))
    print("ASL Citizen aspect", round(a_c, 3), "PopSign estimated aspect", round(a_p, 3), flush=True)
    scale = a_p / a_c
    model, classes = load_model()
    X, y, pid = [], [], []
    for i, r in enumerate(meta.itertuples()):
        lh, rh, pose, face = from_kaggle(pd.read_parquet(io.BytesIO(z.read(r.path))))
        for a in (lh, rh, pose, face):
            a[..., 0] = a[..., 0] * scale
        X.append(normalize(lh, rh, pose, face))
        y.append(classes.index(k2v[r.sign]))
        pid.append(r.participant_id)
        if (i + 1) % 4000 == 0:
            print(i + 1, flush=True)
    X = torch.tensor(np.stack(X))
    y = np.array(y)
    np.savez_compressed(ROOT / "cache/s2r2/kaggle_feats_aspect.npz", X=X.numpy(), y=y, pid=np.array(pid),
                        seq=meta["sequence_id"].to_numpy(), sign=meta["sign"].to_numpy(), scale=scale)
    with torch.no_grad():
        lg = torch.cat([model(X[i:i + 512]) for i in range(0, len(X), 512)])
    top = lg.topk(3, 1).indices.numpy()
    t1, t3 = top[:, 0] == y, (top == y[:, None]).any(1)
    res = {"citizen_aspect_median": a_c, "citizen_face_ratio_px": ratio_px, "popsign_face_ratio_norm": float(np.nanmedian(r_pop)),
           "popsign_aspect_estimated": a_p, "x_scale_applied": scale, "n": int(len(y)), "signers": int(len(set(pid))),
           "top1": acc_with_ci(t1, np.array(pid)), "top3": acc_with_ci(t3, np.array(pid)),
           "round1_top1": 0.3373, "round1_top3": 0.4984}
    jdump(res, "r1_aspect.json")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
