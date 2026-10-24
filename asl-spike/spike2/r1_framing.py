"""R1 follow-up: framing/scale differences that could explain the remaining PopSign gap.

Per clip (2,000 PopSign sequences sampled with seed 0, all ASL Citizen clips of the overlapping classes):
  face_w        face width (landmarks 234-454) as a fraction of image width
  sw_over_face  shoulder width / face width (the round-1 features scale by shoulder width)
  shoulders_out fraction of frames whose shoulder landmarks lie outside the image (y > 1 or x outside 0..1)
  hand_over_face  dominant-hand size (wrist to middle MCP) / face width
"""
from __future__ import annotations

import io
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, citizen_class_map, citizen_rows, jdump, model_classes, vocab  # noqa: E402
from handstats import _motion  # noqa: E402


def stats(lh, rh, pose, face_l, face_r):
    fw = np.abs(face_r[:, 0] - face_l[:, 0])
    sw = np.linalg.norm(pose[:, 11, :2] - pose[:, 12, :2], axis=-1)
    sh = pose[:, 11:13, :2]
    out = ((sh[..., 1] > 1) | (sh[..., 0] < 0) | (sh[..., 0] > 1)).any(-1)
    dom = lh if _motion(lh) > _motion(rh) else rh
    hs = np.linalg.norm(dom[:, 9, :2] - dom[:, 0, :2], axis=-1)
    with np.errstate(all="ignore"):
        return {"face_w": float(np.nanmedian(fw)), "sw_over_face": float(np.nanmedian(sw / fw)),
                "shoulders_out": float(np.nanmean(out)), "hand_over_face": float(np.nanmedian(hs / fw))}


def main():
    classes = model_classes()
    v = {e["gloss"]: e for e in vocab()}
    k2v = {k: g for g in classes for k in v[g]["kaggle_asl_signs"]}
    meta = pd.read_csv(ROOT / "data/kaggle-asl-signs/train.csv")
    meta = meta[meta["sign"].isin(k2v)].sample(2000, random_state=0)
    z = zipfile.ZipFile(ROOT / "data/kaggle-asl-signs/asl-signs.zip")
    rows = []
    for r in meta.itertuples():
        df = pd.read_parquet(io.BytesIO(z.read(r.path)))
        frames = np.sort(df["frame"].unique())
        fi = {f: i for i, f in enumerate(frames)}

        def grab(t, n):
            a = np.full((len(frames), n, 3), np.nan, np.float32)
            s = df[df["type"] == t]
            a[s["frame"].map(fi).to_numpy(), s["landmark_index"].to_numpy()] = s[["x", "y", "z"]].to_numpy(np.float32)
            return a
        face = grab("face", 468)
        rows.append({"dataset": "popsign", "signer": str(r.participant_id),
                     **stats(grab("left_hand", 21), grab("right_hand", 21), grab("pose", 33), face[:, 234], face[:, 454])})
    cmap = citizen_class_map()
    cr = citizen_rows()
    cr = cr[cr["Gloss"].map(cmap).isin(set(k2v.values()))]
    for r in cr.itertuples():
        with np.load(ROOT / "cache/s2/citizen" / f"{os.path.splitext(r.video)[0]}.npz") as d:
            # FACE_SUBSET positions 22, 23 = mesh points 234, 454
            rows.append({"dataset": "citizen", "signer": r.signer, **stats(d["lh"], d["rh"], d["pose"], d["face"][:, 22], d["face"][:, 23])})
    df = pd.DataFrame(rows)
    summ = {ds: {k: {"median": round(float(g[k].median()), 3), "p10": round(float(g[k].quantile(0.1)), 3),
                     "p90": round(float(g[k].quantile(0.9)), 3)} for k in ("face_w", "sw_over_face", "shoulders_out", "hand_over_face")}
            | {"n": len(g), "signers": int(g["signer"].nunique())} for ds, g in df.groupby("dataset")}
    jdump(summ, "r1_framing.json")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
