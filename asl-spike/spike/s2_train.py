"""S2 training/eval on the ASL Citizen vocabulary subset with the official signer-independent splits
(train / val / test have disjoint signers). Early stopping on val top-1; test reported once at the end.
Saves cache/s2/model.pt, results/s2_citizen_eval.json (top-1/3/5, per-sign accuracy, confusions)."""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from features import from_holistic, normalize  # noqa: E402
from s2_model import SignNet, augment, n_params, topk_acc  # noqa: E402
from stats import wilson  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEED = 0


def class_map(vocab) -> dict[str, str]:
    """ASL Citizen gloss -> vocab gloss (prefer the entry where it is an exact name match)."""
    return {g: e["gloss"] for e in vocab for g in e["asl_citizen_class_videos"]}


def load_features():
    vocab = json.load(open(ROOT / "vocab.json"))
    cmap = class_map(vocab)
    rows = pd.read_csv(ROOT / "results/s2_citizen_subset.csv").rename(columns={"Participant ID": "signer", "Video file": "video"})
    X, y, split, signer, vid = [], [], [], [], []
    for r in rows.itertuples():
        p = ROOT / "cache/s2/citizen" / f"{Path(r.video).stem}.npz"
        if not p.exists():
            continue
        d = dict(np.load(p))
        X.append(normalize(*from_holistic(d)))
        y.append(cmap[r.Gloss])
        split.append(r.split)
        signer.append(r.signer)
        vid.append(r.video)
    classes = sorted(set(y))
    ci = {c: i for i, c in enumerate(classes)}
    return np.stack(X), np.array([ci[c] for c in y]), np.array(split), np.array(signer), np.array(vid), classes


KAGGLE_TEST_SIGNERS = 6  # PopSign/Kaggle signers held out when Kaggle data is added to training


def kaggle_split(classes):
    """Kaggle sequences mapped to model classes; a seeded subset of signers is held out for testing."""
    vocab = {e["gloss"]: e for e in json.load(open(ROOT / "vocab.json"))}
    k2v = {k: g for g in classes for k in vocab[g]["kaggle_asl_signs"]}
    F = dict(np.load(ROOT / "cache/s2/kaggle_feats.npz", allow_pickle=True))
    keep = np.array([k in k2v for k in F["sign"]])
    X = F["X"][keep]
    y = np.array([classes.index(k2v[k]) for k in F["sign"][keep]])
    pid = F["pid"][keep]
    ps = sorted(set(pid))
    test_p = set(np.random.default_rng(SEED).choice(ps, KAGGLE_TEST_SIGNERS, replace=False).tolist())
    te = np.array([p in test_p for p in pid])
    return X, y, pid, te, sorted(test_p)


def main(epochs: int = 60, with_kaggle: bool = False):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    t0 = time.time()
    X, y, split, signer, vid, classes = load_features()
    print("features", X.shape, "classes", len(classes), f"{time.time() - t0:.0f}s")
    tr, va, te = (split == "train"), (split == "val"), (split == "test")
    for name, m in (("train", tr), ("val", va), ("test", te)):
        print(name, int(m.sum()), "videos", len(set(signer[m])), "signers")
    model = SignNet(len(classes))
    print("params", n_params(model))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.05)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    Xtr, ytr = X[tr], y[tr]
    kag = None
    if with_kaggle:
        KX, Ky, Kp, Kte, test_p = kaggle_split(classes)
        Xtr, ytr = np.concatenate([Xtr, KX[~Kte]]), np.concatenate([ytr, Ky[~Kte]])
        kag = (KX[Kte], Ky[Kte], test_p, int((~Kte).sum()))
        print("added Kaggle train sequences", int((~Kte).sum()), "held-out Kaggle signers", test_p)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * ((len(Xtr) + 63) // 64))
    Xva, yva = torch.tensor(X[va]), torch.tensor(y[va])
    best, best_state, hist = -1.0, None, []
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(Xtr))
        tl = 0.0
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            xb = torch.tensor(np.stack([augment(Xtr[j], rng) for j in b]))
            loss = lossf(model(xb), torch.tensor(ytr[b]))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tl += float(loss) * len(b)
        model.eval()
        with torch.no_grad():
            acc = topk_acc(model(Xva), yva)
        hist.append({"epoch": ep, "loss": tl / len(perm), **acc})
        if acc["top1"] > best:
            best, best_state = acc["top1"], {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"ep {ep} loss {tl / len(perm):.3f} val {acc} {time.time() - t0:.0f}s", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X[te]))
    yt = torch.tensor(y[te])
    res = {"classes": classes, "n_classes": len(classes), "params": n_params(model), "epochs": epochs, "seed": SEED,
           "best_val_top1": best, "history": hist, "train_s": time.time() - t0,
           "n": {s: {"videos": int(m.sum()), "signers": len(set(signer[m]))} for s, m in (("train", tr), ("val", va), ("test", te))},
           "test": topk_acc(logits, yt)}
    k = int(res["test"]["top1"] * len(yt))
    res["test"]["top1_wilson"] = wilson(k, len(yt))
    pred = logits.argmax(1).numpy()
    per = defaultdict(lambda: [0, 0])
    conf = Counter()
    for t, p in zip(y[te], pred):
        per[classes[t]][0] += int(t == p)
        per[classes[t]][1] += 1
        if t != p:
            conf[(classes[t], classes[p])] += 1
    res["per_sign"] = {c: {"acc": round(a / n, 3), "n": n} for c, (a, n) in sorted(per.items())}
    pairs = Counter()
    for (a, b), n in conf.items():
        pairs[tuple(sorted((a, b)))] += n
    res["top_confused_pairs"] = [{"pair": list(p), "count": n} for p, n in pairs.most_common(15)]
    res["test_predictions"] = [{"video": v, "true": classes[t], "pred": classes[p], "top5": [classes[i] for i in row]}
                               for v, t, p, row in zip(vid[te], y[te], pred, logits.topk(5, 1).indices.numpy())]
    # held-out Kaggle signers (same signers for both variants, so the two models are comparable)
    try:
        KX, Ky, Kp, Kte, test_p = kaggle_split(classes)
        with torch.no_grad():
            kl = torch.cat([model(torch.tensor(KX[Kte][i:i + 512])) for i in range(0, int(Kte.sum()), 512)])
        res["kaggle_heldout_signers"] = {"signers": [str(p) for p in test_p], "n": int(Kte.sum()),
                                         **topk_acc(kl, torch.tensor(Ky[Kte]))}
    except FileNotFoundError:
        pass
    if kag:
        res["kaggle_train_sequences_added"] = kag[3]
    tag = "_with_kaggle" if with_kaggle else ""
    torch.save({"state": model.state_dict(), "classes": classes}, ROOT / f"cache/s2/model{tag}.pt")
    json.dump(res, open(ROOT / f"results/s2_citizen_eval{tag}.json", "w"), indent=1)
    print("TEST", res["test"], "n", len(yt), "kaggle held-out", res.get("kaggle_heldout_signers", {}).get("top1"))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60, with_kaggle="--with-kaggle" in sys.argv)
