"""R1 data pass: hand statistics for every Kaggle (PopSign) sequence of the overlapping signs and every
ASL Citizen clip of the same classes. Writes cache/s2r2/r1_handstats.jsonl (one row per clip)."""
from __future__ import annotations

import io
import json
import os
import sys
import zipfile
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE2, ROOT, citizen_class_map, citizen_rows, model_classes, vocab  # noqa: E402
from handstats import hand_stats, kaggle_arrays  # noqa: E402

KZ = ROOT / "data/kaggle-asl-signs/asl-signs.zip"
_z = None


def kag_one(r):
    global _z
    if _z is None:
        _z = zipfile.ZipFile(KZ)
    path, pid, seq, sign, cls = r
    try:
        df = pd.read_parquet(io.BytesIO(_z.read(path)))
        lh, rh, pose = kaggle_arrays(df)
        return {"dataset": "popsign", "id": int(seq), "signer": str(pid), "kaggle_sign": sign, "class": cls, **hand_stats(lh, rh, pose)}
    except Exception as e:
        return {"dataset": "popsign", "id": int(seq), "signer": str(pid), "class": cls, "error": str(e)[:200]}


def cit_one(r):
    video, signer, split, cls = r
    p = ROOT / "cache/s2/citizen" / f"{os.path.splitext(video)[0]}.npz"
    try:
        with np.load(p) as d:
            return {"dataset": "citizen", "id": video, "signer": signer, "split": split, "class": cls,
                    **hand_stats(d["lh"], d["rh"], d["pose"])}
    except Exception as e:
        return {"dataset": "citizen", "id": video, "signer": signer, "class": cls, "error": str(e)[:200]}


def main():
    classes = model_classes()
    v = {e["gloss"]: e for e in vocab()}
    k2v = {}
    for g in classes:
        for k in v[g]["kaggle_asl_signs"]:
            k2v.setdefault(k, g)
    meta = pd.read_csv(ROOT / "data/kaggle-asl-signs/train.csv")
    meta = meta[meta["sign"].isin(k2v)]
    kjobs = [(r.path, r.participant_id, r.sequence_id, r.sign, k2v[r.sign]) for r in meta.itertuples()]
    cmap = citizen_class_map()
    rows = citizen_rows()
    rows = rows[rows["Gloss"].map(cmap).isin(set(k2v.values()))]
    cjobs = [(r.video, r.signer, r.split, cmap[r.Gloss]) for r in rows.itertuples()]
    out = CACHE2 / "r1_handstats.jsonl"
    with Pool(7) as pool, open(out, "w", encoding="utf-8") as f:
        for i, rec in enumerate(pool.imap_unordered(kag_one, kjobs, chunksize=32)):
            f.write(json.dumps(rec) + "\n")
            if (i + 1) % 4000 == 0:
                print("popsign", i + 1, "of", len(kjobs), flush=True)
        for rec in pool.imap_unordered(cit_one, cjobs, chunksize=16):
            f.write(json.dumps(rec) + "\n")
    print("done", len(kjobs), "popsign,", len(cjobs), "citizen")


if __name__ == "__main__":
    main()
