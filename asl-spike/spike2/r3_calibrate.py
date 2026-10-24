"""R3: calibration curve - how much must the demo signer record?

Stand-ins for the demo signer: the 6 PopSign signers held out in round 1 (53 overlapping signs). A separate
PopSign signer (not among the 6) is the validation signer used to tune fine-tune epochs/LR and the prototype
mixing weight. For each test signer and k in {0,1,2,3,5}: k calibration clips per sign are drawn from that
signer, the rest of that signer's clips are the test set; 3 random draws (= seeds 0, 1, 2).

Methods
  (a) fine-tune: a few epochs on the calibration clips mixed 1:1 with replayed ASL Citizen training clips,
      AdamW, low LR; epochs/LR tuned on the validation signer at k = 3.
  (b) prototypes: penultimate (attention-pooled) embedding averaged per sign over the calibration clips;
      score = (1-a) * log_softmax(base logits) + a * T * cos(embedding, prototype) for calibrated classes,
      base log-probs alone for the others; a tuned on the validation signer (T = 10).
  (c) partial vocabulary: calibrate on only 10 or 25 signs (k = 3), test on the other signs (fine-tune and
      prototypes).
Usage: python r3_calibrate.py <features.npz> <base_model.pt> <tag>
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(__file__))
from common import R2, ROOT, jdump, signer_bootstrap  # noqa: E402
from s2_model import SignNet, augment  # noqa: E402

TEST_SIGNERS = ["2044", "25571", "26734", "29302", "30680", "36257"]
KS = (0, 1, 2, 3, 5)
TEMP = 10.0


def embed(model, X):
    feats = {}
    h = model.head[1].register_forward_hook(lambda m, i, o: feats.__setitem__("e", i[0].detach()))
    with torch.no_grad():
        lg = torch.cat([model(X[i:i + 512]) for i in range(0, len(X), 512)]) if len(X) > 512 else model(X)
    h.remove()
    if len(X) > 512:  # recompute embeddings in chunks
        es = []
        for i in range(0, len(X), 512):
            h = model.head[1].register_forward_hook(lambda m, i_, o: feats.__setitem__("e", i_[0].detach()))
            with torch.no_grad():
                model(X[i:i + 512])
            h.remove()
            es.append(feats["e"])
        return lg, torch.cat(es)
    return lg, feats["e"]


def proto_scores(base_lg, emb, protos, alpha):
    lp = Fn.log_softmax(base_lg, 1)
    if not protos or alpha == 0:
        return lp
    cls = torch.tensor(sorted(protos))
    P = Fn.normalize(torch.stack([protos[c] for c in sorted(protos)]), dim=1)
    sim = Fn.normalize(emb, dim=1) @ P.T
    out = lp.clone()
    out[:, cls] = (1 - alpha) * lp[:, cls] + alpha * TEMP * sim
    return out


def finetune(model, Xc, yc, replay, epochs, lr, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = copy.deepcopy(model)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    RX, Ry = replay
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    n = len(Xc)
    for _ in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, 32):
            b = perm[i:i + 32]
            r = rng.choice(len(RX), len(b), replace=False)
            xb = np.concatenate([np.stack([augment(Xc[j], rng) for j in b]), RX[r]])
            yb = np.concatenate([yc[b], Ry[r]])
            loss = lossf(m(torch.tensor(xb)), torch.tensor(yb))
            opt.zero_grad()
            loss.backward()
            opt.step()
    m.eval()
    return m


def draw(y, k, rng, signs=None):
    cal, test = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        idx = rng.permutation(idx)
        if signs is not None and c not in signs:
            test += list(idx)
            continue
        cal += list(idx[:k])
        test += list(idx[k:])
    return np.array(cal, int), np.array(test, int)


def hits(scores, y):
    top = scores.topk(3, 1).indices.numpy()
    return top[:, 0] == y, (top == y[:, None]).any(1)


def main(feat_path, model_path, tag):
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    F = dict(np.load(feat_path, allow_pickle=True))
    X, y, pid = F["X"].astype(np.float32), F["y"], F["pid"].astype(str)
    ck = torch.load(model_path)
    base = SignNet(len(ck["classes"]))
    base.load_state_dict(ck["state"])
    base.eval()
    # replay: ASL Citizen training features (round-1 pipeline)
    sys.path.insert(0, str(ROOT / "spike"))
    from s2_train import load_features
    CX, Cy, Csplit, *_ = load_features()
    replay = (CX[Csplit == "train"], Cy[Csplit == "train"])
    others = sorted(set(pid) - set(TEST_SIGNERS))
    val_signer = others[int(np.random.default_rng(0).integers(len(others)))]
    # ---- tune on the validation signer at k = 3
    vm = pid == val_signer
    Xv, yv = X[vm], y[vm]
    rng = np.random.default_rng(100)
    cal, te = draw(yv, 3, rng)
    tune = {"val_signer": val_signer, "finetune": [], "proto": []}
    for ep in (3, 6, 10):
        for lr in (1e-4, 3e-4, 1e-3):
            m = finetune(base, Xv[cal], yv[cal], replay, ep, lr, 0)
            with torch.no_grad():
                t1 = float((m(torch.tensor(Xv[te])).argmax(1).numpy() == yv[te]).mean())
            tune["finetune"].append({"epochs": ep, "lr": lr, "val_top1": round(t1, 4)})
    best_ft = max(tune["finetune"], key=lambda g: g["val_top1"])
    lg, em = embed(base, torch.tensor(Xv))
    protos = {int(c): em[cal][yv[cal] == c].mean(0) for c in np.unique(yv[cal])}
    for a in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        t1 = float((proto_scores(lg[te], em[te], protos, a).argmax(1).numpy() == yv[te]).mean())
        tune["proto"].append({"alpha": a, "val_top1": round(t1, 4)})
    best_a = max(tune["proto"], key=lambda g: g["val_top1"])["alpha"]
    print("tuned", best_ft, "alpha", best_a, flush=True)
    # ---- calibration curve on the 6 test signers
    curve = {m_: {k: {"h1": {}, "h3": {}, "secs": []} for k in KS} for m_ in ("finetune", "proto")}
    partial = {m_: {n: {"h1": {}, "h3": {}, "base_h1": {}} for n in (10, 25)} for m_ in ("finetune", "proto")}
    for s in TEST_SIGNERS:
        sm = pid == s
        Xs, ys = X[sm], y[sm]
        lg_all, em_all = embed(base, torch.tensor(Xs))
        for seed in (0, 1, 2):
            rng = np.random.default_rng(1000 * seed + int(s) % 997)
            for k in KS:
                cal, te = draw(ys, k, rng)
                # prototypes
                t0 = time.process_time()
                protos = {int(c): em_all[cal][ys[cal] == c].mean(0) for c in np.unique(ys[cal])} if k else {}
                sc = proto_scores(lg_all[te], em_all[te], protos, best_a if k else 0.0)
                curve["proto"][k]["secs"].append(time.process_time() - t0)
                h1, h3 = hits(sc, ys[te])
                curve["proto"][k]["h1"].setdefault(s, []).extend(h1.tolist())
                curve["proto"][k]["h3"].setdefault(s, []).extend(h3.tolist())
                # fine-tune
                if k:
                    t0 = time.process_time()
                    m = finetune(base, Xs[cal], ys[cal], replay, best_ft["epochs"], best_ft["lr"], seed)
                    curve["finetune"][k]["secs"].append(time.process_time() - t0)
                    with torch.no_grad():
                        h1, h3 = hits(m(torch.tensor(Xs[te])), ys[te])
                else:
                    h1, h3 = hits(lg_all[te], ys[te])
                curve["finetune"][k]["h1"].setdefault(s, []).extend(h1.tolist())
                curve["finetune"][k]["h3"].setdefault(s, []).extend(h3.tolist())
            # partial vocabulary (k = 3 on n signs, test on the others)
            for n in (10, 25):
                signs = set(rng.choice(np.unique(ys), n, replace=False).tolist())
                cal, te = draw(ys, 3, rng, signs)
                te = te[~np.isin(ys[te], list(signs))]
                bh1, _ = hits(lg_all[te], ys[te])
                protos = {int(c): em_all[cal][ys[cal] == c].mean(0) for c in signs}
                ph1, ph3 = hits(proto_scores(lg_all[te], em_all[te], protos, best_a), ys[te])
                m = finetune(base, Xs[cal], ys[cal], replay, best_ft["epochs"], best_ft["lr"], seed)
                with torch.no_grad():
                    fh1, fh3 = hits(m(torch.tensor(Xs[te])), ys[te])
                for m_, a1, a3 in (("proto", ph1, ph3), ("finetune", fh1, fh3)):
                    partial[m_][n]["h1"].setdefault(s, []).extend(a1.tolist())
                    partial[m_][n]["h3"].setdefault(s, []).extend(a3.tolist())
                    partial[m_][n]["base_h1"].setdefault(s, []).extend(bh1.tolist())
        print("signer", s, "done", flush=True)

    def summ(d):
        pooled = np.concatenate([np.asarray(v, float) for v in d.values()]) if d else np.array([])
        return {"acc": round(float(pooled.mean()), 4) if len(pooled) else None, "ci": signer_bootstrap(d), "n": int(len(pooled)), "signers": len(d)}
    # recording time: median PopSign clip length
    hs = [json.loads(x) for x in open(ROOT / "cache/s2r2/r1_handstats.jsonl") if '"popsign"' in x]
    clip_s = float(np.median([h["frames"] for h in hs if "frames" in h])) / 30.0
    n_signs = len(np.unique(y))
    out = {"tag": tag, "features": str(feat_path), "base_model": str(model_path), "tuning": tune,
           "chosen": {"finetune": best_ft, "proto_alpha": best_a, "proto_temperature": TEMP},
           "test_signers": TEST_SIGNERS, "n_signs": int(n_signs), "draws": 3,
           "curve": {m_: {str(k): {"top1": summ(v["h1"]), "top3": summ(v["h3"]),
                                   "cpu_s": {"median": round(float(np.median(v["secs"])), 2) if v["secs"] else 0.0,
                                             "max": round(float(np.max(v["secs"])), 2) if v["secs"] else 0.0},
                                   "recording_min_estimate": round(k * n_signs * clip_s / 60, 1)} for k, v in c.items()}
                     for m_, c in curve.items()},
           "partial_vocab": {m_: {str(n): {"top1_uncalibrated_signs": summ(v["h1"]), "top3_uncalibrated_signs": summ(v["h3"]),
                                           "base_top1_same_items": summ(v["base_h1"])} for n, v in c.items()} for m_, c in partial.items()},
           "popsign_median_clip_s": round(clip_s, 2),
           "note": "recording time = k x signs x median PopSign clip length (Kaggle frames / 30 fps); excludes setup and retakes"}
    jdump(out, f"r3_calibration_{tag}.json")
    for m_ in ("proto", "finetune"):
        print(m_, {k: (out["curve"][m_][k]["top1"]["acc"], out["curve"][m_][k]["top3"]["acc"], out["curve"][m_][k]["cpu_s"]["median"]) for k in out["curve"][m_]})
    print("partial", {m_: {n: (v["top1_uncalibrated_signs"]["acc"], v["base_top1_same_items"]["acc"]) for n, v in out["partial_vocab"][m_].items()} for m_ in out["partial_vocab"]})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
