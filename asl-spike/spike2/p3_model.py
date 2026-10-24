"""P3: phrase-mode CTC model.

Input per frame: the 114 body-coordinate features from p2_gen (+ frame-to-frame velocity), whole sentence,
30 fps. Encoder as round 1 (Linear 228->192, LayerNorm, GELU; 3 residual Conv1d k=5 blocks; GRU), head
Linear -> 145 = 143 classes + <unk> (143) + blank (144). Initialized from the round-1 isolated model where
shapes match.
  bi:      bidirectional GRU (96 per direction), centered convolutions (offline: needs the whole utterance)
  stream:  unidirectional GRU (192), causal convolutions (no lookahead)
Training batches are generated on the fly by p2_gen (train-split signers only), in DataLoader workers.
Usage: python p3_model.py <bi|stream> <minutes> <tag> [--rwth-fold K] [--rwth-calib] [--init path]
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE2, R2, ROOT, model_classes  # noqa: E402

N_CLS = 143
UNK_ID, BLANK = 143, 144


class Block(nn.Module):
    def __init__(self, d, causal):
        super().__init__()
        self.causal = causal
        self.conv = nn.Conv1d(d, d, 5, padding=0)
        self.bn = nn.BatchNorm1d(d)
        self.act = nn.GELU()

    def forward(self, h):  # (B, d, T)
        h2 = nn.functional.pad(h, (4, 0) if self.causal else (2, 2))
        return h + self.act(self.bn(self.conv(h2)))


class PhraseNet(nn.Module):
    def __init__(self, mode="bi", d=192):
        super().__init__()
        self.mode = mode
        self.inp = nn.Sequential(nn.Linear(228, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(0.1))
        self.conv = nn.ModuleList([Block(d, mode == "stream") for _ in range(3)])
        self.gru = nn.GRU(d, d // 2 if mode == "bi" else d, batch_first=True, bidirectional=(mode == "bi"))
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(d, N_CLS + 2))

    def forward(self, x):  # x: (B, T, 114)
        v = torch.diff(x, dim=1, prepend=x[:, :1])
        h = self.inp(torch.cat([x, v], -1)).transpose(1, 2)
        for c in self.conv:
            h = c(h)
        h, _ = self.gru(h.transpose(1, 2))
        return self.head(h)  # (B, T, 145) logits


def init_from_round1(net: PhraseNet):
    sd = torch.load(ROOT / "cache/s2/model.pt")["state"]
    own = net.state_dict()
    n = 0
    for k, v in sd.items():
        k2 = k.replace("conv.0.0.", "conv.0.conv.").replace("conv.1.0.", "conv.1.conv.").replace("conv.2.0.", "conv.2.conv.")
        k2 = k2.replace("conv.0.1.", "conv.0.bn.").replace("conv.1.1.", "conv.1.bn.").replace("conv.2.1.", "conv.2.bn.")
        if k2 in own and own[k2].shape == v.shape:
            own[k2] = v
            n += 1
    net.load_state_dict(own)
    return n


def label_ids(labels, classes):
    return [UNK_ID if l == "<unk>" else classes.index(l) for l in labels]


class SynthStream(torch.utils.data.IterableDataset):
    def __init__(self, seed, batch=16):
        self.seed, self.batch = seed, batch

    def __iter__(self):
        import p2_gen as G
        wi = torch.utils.data.get_worker_info()
        pools = G.Pools()
        g = G.Generator(pools, "train", self.seed * 1000 + (wi.id if wi else 0))
        classes = pools.classes
        while True:
            items = [g.sentence() for _ in range(self.batch)]
            yield collate([(X, label_ids(l, classes)) for X, l, _ in items])


def collate(items):
    T = max(len(x) for x, _ in items)
    X = np.zeros((len(items), T, 114), np.float32)
    lens = []
    for i, (x, _) in enumerate(items):
        X[i, :len(x)] = x
        lens.append(len(x))
    tgt = [torch.tensor(y, dtype=torch.long) for _, y in items]
    return torch.tensor(X), torch.tensor(lens), torch.cat(tgt) if tgt else torch.zeros(0, dtype=torch.long), torch.tensor([len(t) for t in tgt])


def rwth_items(fold=None, calib=False, part="train"):
    """RWTH-BOSTON-104 sentences as (features, label ids, signer). fold = held-out signer index (0..2).
    part: 'train' = the other signers (+ calib half of the held-out signer if calib), 'test' = held-out signer
    (the non-calibration half if calib)."""
    import p2_gen as G
    from s2_continuous import alias_map, norm
    classes = model_classes()
    amap = alias_map(classes)
    rows = json.load(open(ROOT / "results/s3_rwth_ncslgr_match.json"))["rows"]
    signers = sorted({r["speaker"] for r in rows})
    out = []
    for r in rows:
        p = ROOT / "cache/s3/rwth" / f"{r['name']}.npz"
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        sc = (float(d["width"]) / float(d["height"])) / G.CIT_ASPECT
        from extract import FACE_SUBSET
        face6 = d["face"][:, [FACE_SUBSET[i] for i in G.FACE6]]
        c, _ = G.canon(d["lh"], d["rh"], d["pose"][:, G.POSE8], face6, sc)
        fps = float(d["fps"]) or 25.0
        if abs(fps - G.FPS) > 1:
            c = G.resample(c, fps / G.FPS)
        labels = [amap.get(norm(t), "<unk>") for t in r["orth"].split()]
        out.append({"name": r["name"], "X": G.to_feat(c), "labels": labels, "signer": r["speaker"],
                    "n_vocab": sum(l != "<unk>" for l in labels)})
    if fold is None:
        return out, signers
    held = signers[fold]
    rng = np.random.default_rng(fold)
    hs = [o for o in out if o["signer"] == held]
    perm = rng.permutation(len(hs))
    calib_set = {hs[i]["name"] for i in perm[: len(hs) // 2]} if calib else set()
    if part == "train":
        return [o for o in out if o["signer"] != held or o["name"] in calib_set], signers
    return [o for o in out if o["signer"] == held and o["name"] not in calib_set], signers


def train(mode, minutes, tag, init=None, rwth_fold=None, rwth_calib=False, seed=0, lr=1e-3):
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    net = PhraseNet(mode)
    n_init = init_from_round1(net) if init is None else net.load_state_dict(torch.load(init)["state"]) or -1
    classes = model_classes()
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    dl = torch.utils.data.DataLoader(SynthStream(seed), batch_size=None, num_workers=3, persistent_workers=True)
    real = []
    if rwth_fold is not None:
        real, _ = rwth_items(rwth_fold, rwth_calib, "train")
        real = [(o["X"], label_ids(o["labels"], classes)) for o in real]
    rng = np.random.default_rng(seed)
    t0, step, hist = time.time(), 0, []
    net.train()
    for X, lens, tgt, tl in dl:
        if real and rng.random() < 0.5:  # half the batches: real RWTH sentences
            b = [real[i] for i in rng.choice(len(real), min(16, len(real)), replace=False)]
            X, lens, tgt, tl = collate(b)
        lp = net(X).log_softmax(-1).transpose(0, 1)
        loss = ctc(lp, tgt, lens, tl)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 50 == 0:
            hist.append({"step": step, "loss": round(float(loss), 4), "s": round(time.time() - t0)})
            print(hist[-1], flush=True)
        if time.time() - t0 > minutes * 60:
            break
    torch.save({"state": net.state_dict(), "mode": mode}, CACHE2 / f"p3_{tag}.pt")
    meta = {"tag": tag, "mode": mode, "minutes": minutes, "train_s": round(time.time() - t0), "steps": step,
            "params": sum(p.numel() for p in net.parameters()), "init_tensors_from_round1": n_init, "init": init,
            "rwth_fold": rwth_fold, "rwth_calib": rwth_calib, "seed": seed, "lr": lr, "history": hist}
    json.dump(meta, open(R2 / f"p3_train_{tag}.json", "w"), indent=1)
    print("saved", tag, meta["train_s"], "s", step, "steps")


if __name__ == "__main__":
    a = sys.argv
    kw = {}
    if "--rwth-fold" in a:
        kw["rwth_fold"] = int(a[a.index("--rwth-fold") + 1])
    if "--rwth-calib" in a:
        kw["rwth_calib"] = True
    if "--init" in a:
        kw["init"] = a[a.index("--init") + 1]
    if "--lr" in a:
        kw["lr"] = float(a[a.index("--lr") + 1])
    train(a[1], float(a[2]), a[3], **kw)
