"""R4: robust training ablation for the isolated-sign model (same SignNet as round 1, ~0.8M params).

Variants (3 seeds each):
  baseline   ASL Citizen train, round-1 augmentation
  aug        ASL Citizen train, stronger augmentation: rotation +-15 deg, scale +-20%, translation +-0.15,
             temporal speed 0.7-1.4x, frame dropout 10%, landmark jitter (sd 0.01), hand-slot swap + mirror 10%
  wlasl      baseline data + WLASL training-split signers (R2 split), round-1 augmentation
  popsign    baseline data + PopSign training signers (one-handed selfie style!), round-1 augmentation
  combined   all three datasets + stronger augmentation
Features: round-1 pipeline; WLASL and PopSign with the R1 aspect fix (x rescaled to ASL Citizen's 4:3 pixel geometry).
Steps: baseline = 60 epochs of ASL Citizen; larger variants get min(60 epochs, 2.5x baseline steps).
Checkpoint selection: best mean top-1 over validation signers of all three datasets (ASL Citizen val split,
WLASL val signers, one PopSign validation signer). Test: ASL Citizen test, WLASL test signers, the 6 held-out
PopSign signers. Usage: python r4_train.py <variant> <seed>
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE2, R2, ROOT, jdump  # noqa: E402
from features import N_FEAT, T  # noqa: E402
from s2_model import SignNet, augment, topk_acc  # noqa: E402

POP_TEST = ["2044", "25571", "26734", "29302", "30680", "36257"]
XY_END = N_FEAT - 2


def strong_augment(x, rng):
    x = x.copy()
    pts = x[:, :XY_END].reshape(T, -1, 2)
    present = np.abs(pts).sum(-1) > 0
    s = rng.uniform(0.8, 1.2)
    a = math.radians(rng.uniform(-15, 15))
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], np.float32)
    pts = (pts @ R.T) * s + rng.uniform(-0.15, 0.15, 2).astype(np.float32)
    pts = pts + rng.normal(0, 0.01, pts.shape).astype(np.float32)
    pts[~present] = 0
    if rng.random() < 0.1:  # hand-slot swap + mirror (dominance detection errors / left-handed signers)
        pts[..., 0] = -pts[..., 0]
        pts[:, :21], pts[:, 21:42] = pts[:, 21:42].copy(), pts[:, :21].copy()
        fl = x[:, -2:].copy()
        x[:, -2], x[:, -1] = fl[:, 1], fl[:, 0]
    x[:, :XY_END] = pts.reshape(T, -1)
    # temporal speed 0.7-1.4x: take a window of T/speed source frames (clamped) and resample to T
    sp = rng.uniform(0.7, 1.4)
    L = min(T - 1, max(4, (T - 1) / sp))
    st = rng.uniform(0, T - 1 - L) if T - 1 - L > 0 else 0
    src = np.clip(np.linspace(st, st + L, T), 0, T - 1)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    w = (src - i0)[:, None]
    x = x[i0] * (1 - w) + x[i1] * w
    for t in range(1, T):
        if rng.random() < 0.10:
            x[t] = x[t - 1]
    return x.astype(np.float32)


def load_all():
    cache = CACHE2 / "r4_data.npz"
    if cache.exists():
        return dict(np.load(cache, allow_pickle=True))
    sys.path.insert(0, str(ROOT / "spike"))
    from s2_train import load_features
    from r2_eval import load_items, wl_features
    import torch as _t
    CX, Cy, Csplit, Csig, _, classes = load_features()
    items, _ = load_items()
    split = json.load(open(R2 / "wlasl_split.json"))["items"]
    WX = np.stack([wl_features(it["d"], True) for it in items])
    Wy = np.array([classes.index(it["class"]) for it in items])
    Wsplit = np.array([split[it["video_id"]] for it in items])
    Wsig = np.array([it["signer"] for it in items])
    P = dict(np.load(CACHE2 / "kaggle_feats_aspect.npz", allow_pickle=True))
    Psig = P["pid"].astype(str)
    others = sorted(set(Psig) - set(POP_TEST))
    pval = others[int(np.random.default_rng(0).integers(len(others)))]  # same validation signer as R3
    Psplit = np.where(np.isin(Psig, POP_TEST), "test", np.where(Psig == pval, "val", "train"))
    out = {"CX": CX, "Cy": Cy, "Csplit": Csplit, "Csig": Csig, "WX": WX, "Wy": Wy, "Wsplit": Wsplit, "Wsig": Wsig,
           "PX": P["X"].astype(np.float32), "Py": P["y"], "Psplit": Psplit, "Psig": Psig, "classes": np.array(classes)}
    np.savez_compressed(cache, **out)
    return out


def evaluate(model, X, y):
    if len(X) == 0:
        return None
    with torch.no_grad():
        lg = torch.cat([model(torch.tensor(X[i:i + 512])) for i in range(0, len(X), 512)])
    return topk_acc(lg, torch.tensor(y))


def main(variant, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    D = load_all()
    classes = list(D["classes"])
    tr = D["Csplit"] == "train"
    Xs, ys = [D["CX"][tr]], [D["Cy"][tr]]
    if variant in ("wlasl", "combined"):
        m = D["Wsplit"] == "train"
        Xs.append(D["WX"][m]); ys.append(D["Wy"][m])
    if variant in ("popsign", "combined"):
        m = D["Psplit"] == "train"
        Xs.append(D["PX"][m]); ys.append(D["Py"][m])
    Xtr, ytr = np.concatenate(Xs), np.concatenate(ys)
    aug = strong_augment if variant in ("aug", "combined") else augment
    base_steps = 60 * math.ceil(int(tr.sum()) / 64)
    steps = min(60 * math.ceil(len(Xtr) / 64), int(2.5 * base_steps))
    model = SignNet(len(classes))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    vals = {"citizen": (D["CX"][D["Csplit"] == "val"], D["Cy"][D["Csplit"] == "val"]),
            "wlasl": (D["WX"][D["Wsplit"] == "val"], D["Wy"][D["Wsplit"] == "val"]),
            "popsign": (D["PX"][D["Psplit"] == "val"], D["Py"][D["Psplit"] == "val"])}
    best, best_state, hist, t0 = -1, None, [], time.time()
    step, eval_every = 0, max(1, steps // 30)
    while step < steps:
        perm = rng.permutation(len(Xtr))
        for i in range(0, len(perm), 64):
            if step >= steps:
                break
            b = perm[i:i + 64]
            xb = torch.tensor(np.stack([aug(Xtr[j], rng) for j in b]))
            loss = lossf(model(xb), torch.tensor(ytr[b]))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % eval_every == 0 or step == steps:
                model.eval()
                v = {k: evaluate(model, *xy)["top1"] for k, xy in vals.items()}
                model.train()
                score = float(np.mean(list(v.values())))
                hist.append({"step": step, "loss": round(float(loss), 4), **{f"val_{k}": round(a, 4) for k, a in v.items()}})
                if score > best:
                    best, best_state = score, {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    tests = {"citizen": (D["CX"][D["Csplit"] == "test"], D["Cy"][D["Csplit"] == "test"]),
             "wlasl": (D["WX"][D["Wsplit"] == "test"], D["Wy"][D["Wsplit"] == "test"]),
             "popsign": (D["PX"][D["Psplit"] == "test"], D["Py"][D["Psplit"] == "test"])}
    res = {"variant": variant, "seed": seed, "steps": steps, "train_n": int(len(Xtr)), "train_s": round(time.time() - t0),
           "best_val_mean_top1": best, "history": hist, "test": {k: evaluate(model, *xy) for k, xy in tests.items()},
           "test_n": {k: int(len(xy[1])) for k, xy in tests.items()}}
    torch.save({"state": model.state_dict(), "classes": classes}, CACHE2 / f"r4_{variant}_s{seed}.pt")
    jdump(res, f"r4_{variant}_s{seed}.json")
    print(variant, seed, res["train_s"], "s", {k: round(v["top1"], 4) for k, v in res["test"].items()}, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
