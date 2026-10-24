"""Train the CTC fingerspelling reader (server/fingerspell_ctc.py) on Google's Kaggle ASL Fingerspelling landmarks.

The Kaggle winners trained large Squeezeformer/Transformer CTC models on all 67k sequences with GPUs; this is the same
recipe shrunk to a CPU budget: a few landmark files, ~1M parameters, a time limit. Held out: one whole landmark file
(--holdout), never trained on: half of its letter phrases pick the best epoch, the other half give the reported
character error rate. Files whose features are cached (*.ctc.npz) are used even if the parquet was deleted.

Data: kaggle competitions download -c asl-fingerspelling -f train.csv / -f train_landmarks/<id>.parquet into
data/kaggle_fs/ (unzipped). Run: uv run --with pyarrow --with pandas python -m training.train_fingerspell_ctc
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from server.fingerspell_ctc import DEFAULT_MODEL, MAX_T, SpellNet, features, greedy

ROOT = Path(__file__).resolve().parent.parent
FS = ROOT / "data" / "kaggle_fs"


def cer(ref: str, hyp: str) -> float:
    d = np.arange(len(hyp) + 1)
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1] / max(len(ref), 1)


def load_file(parquet: Path, meta, cidx: dict[str, int]) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """(features, label ids, phrase) per sequence, cached next to the parquet."""
    cache = parquet.with_suffix(".ctc.npz")
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return list(z["items"])
    import pandas as pd

    cols = [f"{a}_{s}_hand_{i}" for s in ("right", "left") for i in range(21) for a in "xyz"]
    df = pd.read_parquet(parquet, columns=cols)
    m = meta[meta.file_id == int(parquet.stem)].set_index("sequence_id")["phrase"]
    items = []
    for seq, g in df.groupby(level=0):
        if seq not in m.index:
            continue
        a = g.to_numpy(np.float32).reshape(len(g), 2, 21, 3)
        x = features(a[:, 0], a[:, 1])
        phrase = str(m.loc[seq])
        y = np.array([cidx[c] + 1 for c in phrase if c in cidx], np.int64)
        if len(y) and len(x) >= len(y):
            items.append((x, y, phrase))
    np.savez(cache, items=np.array(items, dtype=object))
    return items


def augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = x.copy()
    coords = x[:, :-2].reshape(len(x), 2, 21, 3)
    th = rng.normal(0, 0.15)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], np.float32)
    coords[..., :2] = coords[..., :2] @ rot.T
    coords *= rng.uniform(0.85, 1.15)
    x[:, :-2] = coords.reshape(len(x), -1)
    n = len(x)
    keep = np.sort(rng.choice(n, size=max(2, int(n * rng.uniform(0.8, 1.0))), replace=False))  # tempo + dropped frames
    return x[keep]


def batches(items, size: int, rng: np.random.Generator):
    order = np.argsort([len(x) + rng.uniform(0, 8) for x, _, _ in items])  # length buckets, lightly shuffled
    chunks = [order[i:i + size] for i in range(0, len(order), size)]
    rng.shuffle(chunks)
    return chunks


def collate(items, idx, rng=None):
    xs = [augment(items[i][0], rng) if rng is not None else items[i][0] for i in idx]
    xs = [x[np.linspace(0, len(x) - 1, MAX_T).astype(int)] if len(x) > MAX_T else x for x in xs]
    ys = [items[i][1] for i in idx]
    T = max(len(x) for x in xs)
    X = np.zeros((len(xs), T, xs[0].shape[1]), np.float32)
    pad = np.ones((len(xs), T), bool)
    for k, x in enumerate(xs):
        X[k, :len(x)], pad[k, :len(x)] = x, False
    return (torch.from_numpy(X), torch.from_numpy(pad), torch.tensor([len(x) for x in xs]),
            torch.from_numpy(np.concatenate(ys)), torch.tensor([len(y) for y in ys]))


def evaluate(model, items, chars) -> float:
    model.eval()
    scores = []
    with torch.no_grad():
        for x, _, phrase in items:
            x = x[np.linspace(0, len(x) - 1, MAX_T).astype(int)] if len(x) > MAX_T else x
            logp = torch.log_softmax(model(torch.from_numpy(x)[None]), -1)[0].numpy()
            text, _ = greedy(logp, chars)
            scores.append(cer(re.sub(r"[^A-Z]", "", phrase.upper()), re.sub(r"[^A-Z]", "", text.upper())))
    model.train()
    return float(np.mean(scores))


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="1019715464")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--minutes", type=float, default=150)
    ap.add_argument("--threads", type=int, default=12)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cidx = json.loads((FS / "character_to_prediction_index.json").read_text(encoding="utf-8"))
    chars = [c for c, _ in sorted(cidx.items(), key=lambda kv: kv[1])]
    meta = pd.read_csv(FS / "train.csv")
    files = sorted({p.with_name(p.name.replace(".ctc.npz", ".parquet")) for p in FS.glob("*.ctc.npz")} | set(FS.glob("*.parquet")))
    train, test = [], []
    for f in files:
        items = load_file(f, meta, cidx)
        (test if f.stem == a.holdout else train).extend(items)
        print(f"{f.name}: {len(items)} sequences", flush=True)
    test = [t for t in test if len(re.sub(r"[^A-Z]", "", t[2].upper())) >= 0.8 * len(t[2].replace(" ", ""))][:400]
    val, test = test[::2], test[1::2]  # pick the epoch on val; report test, which never influenced any choice
    print(f"train {len(train)} sequences; held-out letter phrases: {len(val)} val, {len(test)} test", flush=True)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    model = SpellNet(len(chars))
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-2)
    steps = a.epochs * ((len(train) + 31) // 32)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1.5e-3, total_steps=steps, pct_start=0.1)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    best, t0 = float("inf"), time.time()
    for ep in range(a.epochs):
        losses = []
        for idx in batches(train, 32, rng):
            X, pad, xl, Y, yl = collate(train, idx, rng)
            logp = torch.log_softmax(model(X, pad), -1).transpose(0, 1)  # (T, B, C)
            loss = ctc(logp, Y, xl, yl)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        score = evaluate(model, val, chars)
        mark = ""
        if score < best:
            best, mark = score, "  (best, saved)"
            DEFAULT_MODEL.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state": model.state_dict(), "chars": chars}, DEFAULT_MODEL)
        print(f"epoch {ep + 1}: loss {np.mean(losses):.3f}  val letter CER {score:.3f}{mark}  "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
        if (time.time() - t0) / 60 > a.minutes:
            print("time budget reached", flush=True)
            break
    model.load_state_dict(torch.load(DEFAULT_MODEL, weights_only=True)["state"])
    print(f"best val letter CER {best:.3f}; held-out TEST letter CER {evaluate(model, test, chars):.3f} -> {DEFAULT_MODEL}")


if __name__ == "__main__":
    main()
