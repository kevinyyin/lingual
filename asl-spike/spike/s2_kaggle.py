"""S2 cross-dataset test: the ASL-Citizen-trained model, without retraining, on the Kaggle asl-signs
landmark files (PopSign signers, phone selfie cameras) for glosses in both our vocabulary and the 250-sign
list. Landmarks are mapped to the same feature layout (features.from_kaggle -> normalize).

Two scorings: (1) argmax over all vocabulary classes the model knows (realistic), (2) argmax restricted to
the overlapping classes (what a 250-sign-overlap demo would see). Results: results/s2_kaggle_eval.json."""
from __future__ import annotations

import io
import json
import os
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from features import from_kaggle, normalize  # noqa: E402
from s2_model import SignNet  # noqa: E402
from stats import wilson  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
KZ = ROOT / "data/kaggle-asl-signs/asl-signs.zip"


def build_features():
    """Features for every Kaggle sequence whose sign is in vocab.json's kaggle matches (cached)."""
    cache = ROOT / "cache/s2/kaggle_feats.npz"
    if cache.exists():
        return dict(np.load(cache, allow_pickle=True))
    vocab = json.load(open(ROOT / "vocab.json"))
    signs = {k for e in vocab if e["asl_citizen_class_videos"] for k in e["kaggle_asl_signs"]}
    meta = pd.read_csv(ROOT / "data/kaggle-asl-signs/train.csv")
    meta = meta[meta["sign"].isin(signs)]
    z = zipfile.ZipFile(KZ)
    X, sg, pid, fails = [], [], [], 0
    for i, r in enumerate(meta.itertuples()):
        try:
            df = pd.read_parquet(io.BytesIO(z.read(r.path)))
            X.append(normalize(*from_kaggle(df)))
            sg.append(r.sign)
            pid.append(r.participant_id)
        except Exception:
            fails += 1
        if (i + 1) % 2000 == 0:
            print(i + 1, "of", len(meta), flush=True)
    out = {"X": np.stack(X), "sign": np.array(sg), "pid": np.array(pid), "fails": fails}
    np.savez_compressed(cache, **out)
    return out


def main(max_per_sign: int | None = None):
    ck = torch.load(ROOT / "cache/s2/model.pt")
    classes = ck["classes"]
    model = SignNet(len(classes))
    model.load_state_dict(ck["state"])
    model.eval()
    vocab = {e["gloss"]: e for e in json.load(open(ROOT / "vocab.json"))}
    k2v = {}
    for g in classes:
        for k in vocab[g]["kaggle_asl_signs"]:
            k2v.setdefault(k, g)
    F = build_features()
    keep = np.array([k in k2v for k in F["sign"]])
    X = torch.tensor(F["X"][keep])
    y = np.array([classes.index(k2v[k]) for k in F["sign"][keep]])
    pid = list(F["pid"][keep])
    fails = int(F["fails"])
    print("overlap signs", len(set(F["sign"][keep])), "sequences", len(y), "signers", len(set(pid)))
    with torch.no_grad():
        logits = torch.cat([model(X[i:i + 512]) for i in range(0, len(X), 512)])
    ov = sorted(set(y))
    mask = torch.full((len(classes),), float("-inf"))
    mask[ov] = 0
    res = {"overlap_vocab_glosses": [classes[i] for i in ov], "n_overlap": len(ov), "n_sequences": int(len(y)),
           "n_signers": int(len(set(pid))), "read_failures": fails, "kaggle_to_vocab": k2v}
    for name, lg in (("all_vocab_classes", logits), ("restricted_to_overlap", logits + mask)):
        top = lg.topk(5, 1).indices.numpy()
        t1 = top[:, 0] == y
        t3 = (top[:, :3] == y[:, None]).any(1)
        per = defaultdict(lambda: [0, 0, 0])
        for yy, a, b in zip(y, t1, t3):
            per[classes[yy]][0] += a
            per[classes[yy]][1] += b
            per[classes[yy]][2] += 1
        res[name] = {"top1": float(t1.mean()), "top3": float(t3.mean()), "top5": float((top == y[:, None]).any(1).mean()),
                     "top1_wilson": wilson(int(t1.sum()), len(y)),
                     "per_sign": {c: {"top1": round(a / n, 3), "top3": round(b / n, 3), "n": n} for c, (a, b, n) in sorted(per.items())},
                     "per_signer_top1": {str(p): round(float(np.mean(t1[np.array(pid) == p])), 3) for p in sorted(set(pid))}}
        print(name, {k: res[name][k] for k in ("top1", "top3", "top5")})
    json.dump(res, open(ROOT / "results/s2_kaggle_eval.json", "w"), indent=1, default=int)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        F = build_features()
        print("built", F["X"].shape, "fails", F["fails"])
    else:
        main()
