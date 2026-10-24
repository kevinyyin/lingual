"""S2: evaluate ASL Citizen's released pose baseline (ST-GCN, checkpoints_v1) on our vocabulary subset of
the official test split, without fine-tuning. Inputs rebuilt from our MediaPipe Holistic landmarks with the
repo's preprocessing (543x2 layout, max 128 frames, shoulder-centered/scaled, 27 keypoints, pose+lh+rh).

Scoring: (1) the model's own 2,731-way top-k (sanity check against the paper), (2) restricted to our vocabulary:
each vocabulary class scores the max logit over its member ASL Citizen glosses (variants such as WANT1/WANT2).
Results: results/s2_stgcn_eval.json."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT / "data/asl-citizen-code/ASL-citizen-code-main/ST-GCN"
sys.path.insert(0, str(REPO))
from architecture.fc import FC  # noqa: E402
from architecture.network import Network  # noqa: E402
from architecture.st_gcn import STGCN  # noqa: E402

KEYPOINTS = [0, 2, 5, 11, 12, 13, 14, 33, 37, 38, 41, 42, 45, 46, 49, 50, 53, 54, 58, 59, 62, 63, 66, 67, 70, 71, 74]
GRAPH = {'num_nodes': 27, 'center': 0,
         'inward_edges': [[2, 0], [1, 0], [0, 3], [0, 4], [3, 5], [4, 6], [5, 7], [6, 17], [7, 8], [7, 9], [9, 10], [7, 11],
                          [11, 12], [7, 13], [13, 14], [7, 15], [15, 16], [17, 18], [17, 19], [19, 20], [17, 21], [21, 22],
                          [17, 23], [23, 24], [17, 25], [25, 26]]}


def downsample(frames, max_frames):
    length = frames.shape[0]
    increment = max_frames / length
    count, new = 0.0, []
    for f in frames:
        count += increment
        if count >= 1:
            count -= 1
            new.append(f)
    return np.array(new[:max_frames])


def prep(d):
    T = len(d["pose"])
    f = np.zeros((T, 75, 2))
    f[:, 0:33] = np.nan_to_num(d["pose"][:, :, :2])
    f[:, 33:54] = np.nan_to_num(d["rh"][:, :, :2])
    f[:, 54:75] = np.nan_to_num(d["lh"][:, :, :2])
    if T > 128:
        f = downsample(f, 128)
    if len(f) < 128:
        f = np.pad(f, ((0, 128 - len(f)), (0, 0), (0, 0)))
    sl, sr = f[:, 11], f[:, 12]
    center = ((sl + sr) / 2).mean(0)
    md = np.mean(np.sqrt(((sl - sr) ** 2).sum(-1)))
    if md != 0:
        f = (f - center) / md
    data = np.concatenate([f[:, 0:33], f[:, 54:], f[:, 33:54]], axis=1)[:, KEYPOINTS]
    return np.transpose(data, (2, 0, 1))


def main():
    torch.set_default_dtype(torch.float64)
    sp = ROOT / "data/asl-citizen/ASL_Citizen/splits"
    train = pd.read_csv(sp / "train.csv")
    glosses = sorted({str(g).strip() for g in train["Gloss"]})
    g2i = {g: i for i, g in enumerate(glosses)}
    model = Network(encoder=STGCN(in_channels=2, graph_args=GRAPH, edge_importance_weighting=True),
                    decoder=FC(n_features=256, num_class=len(glosses), dropout_ratio=0.05))
    model.load_state_dict(torch.load(ROOT / "data/asl-citizen-code/weights/ASL_citizen_stgcn_weights.pt", map_location="cpu"))
    model.eval()
    vocab = json.load(open(ROOT / "vocab.json"))
    cls = [e["gloss"] for e in vocab if e["asl_citizen_class_videos"]]
    members = {e["gloss"]: [g2i[g] for g in e["asl_citizen_class_videos"] if g in g2i] for e in vocab if e["asl_citizen_class_videos"]}
    c2v = {g: e["gloss"] for e in vocab for g in e["asl_citizen_class_videos"]}
    sub = pd.read_csv(ROOT / "results/s2_citizen_subset.csv")
    test = sub[sub["split"] == (sys.argv[1] if len(sys.argv) > 1 else "test")]
    X, yv, yfull, missing = [], [], [], 0
    for r in test.itertuples():
        p = ROOT / "cache/s2/citizen" / f"{Path(r._2).stem}.npz"
        if not p.exists():
            missing += 1
            continue
        with np.load(p) as z:
            X.append(prep(dict(z)))
        yv.append(cls.index(c2v[r.Gloss]))
        yfull.append(g2i[r.Gloss])
    X = torch.tensor(np.stack(X))
    with torch.no_grad():
        logits = torch.cat([model(X[i:i + 64]) for i in range(0, len(X), 64)])
    yfull, yv = np.array(yfull), np.array(yv)
    top = logits.topk(10, 1).indices.numpy()
    res = {"n_test": int(len(yv)), "missing_landmarks": missing,
           "full_2731": {f"top{k}": float((top[:, :k] == yfull[:, None]).any(1).mean()) for k in (1, 5, 10)}}
    vl = torch.stack([logits[:, members[c]].max(1).values for c in cls], 1)
    vt = vl.topk(5, 1).indices.numpy()
    res["vocab_restricted"] = {f"top{k}": float((vt[:, :k] == yv[:, None]).any(1).mean()) for k in (1, 3, 5)}
    res["note"] = ("released checkpoint, no fine-tuning; landmarks from the MediaPipe Tasks Holistic Landmarker "
                   "(the repo used legacy mp.solutions.holistic), so small preprocessing differences are possible")
    res["split"] = sys.argv[1] if len(sys.argv) > 1 else "test"
    json.dump(res, open(ROOT / f"results/s2_stgcn_eval{'' if res['split'] == 'test' else '_' + res['split']}.json", "w"), indent=1)
    print(res)


if __name__ == "__main__":
    main()
