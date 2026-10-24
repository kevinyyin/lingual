"""Train the sign classifier on ASL Citizen (+ our recordings) and save models/signs.pt.

Run: uv run python -m training.train_signs [--train-signers kevin,ana] [--holdout-signer sam] [--epochs 60]
Splits: ASL Citizen train (+ ours except holdout) / ASL Citizen val / ASL Citizen test + holdout signer.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from server.classifier import DEFAULT_MODEL, SignNet, save_checkpoint
from server.features import sequence_features
from server.vocab import load_vocab
from training.landmarks import load_npz

ROOT = Path(__file__).resolve().parent.parent
LM = ROOT / "data" / "landmarks" / "asl_citizen"
OURS = ROOT / "data" / "ours"


def load_dir(root: Path, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i, g in enumerate(labels):
        for p in sorted((root / g).glob("*.npz")):
            frames = load_npz(p)
            if len(frames) >= 4:
                X.append(sequence_features(frames))
                y.append(i)
    if not X:
        return np.zeros((0, 32, 0), np.float32), np.zeros(0, int)
    return np.stack(X), np.array(y)


# Index triples (x, y, z) inside the feature vector; presence flags at 63 and 127 are excluded.
TRIPLES = np.array([[h + 3 * i + k for k in range(3)] for h in (0, 64) for i in range(21)]
                   + [[128 + 3 * i + k for k in range(3)] for i in range(2 + 7)])
SWAP = np.r_[64:128, 0:64, 131:134, 128:131, 134:137, 140:143, 137:140, 146:149, 143:146, 152:155, 149:152]  # swap hands, wrists and pose L/R pairs


def augment(x: torch.Tensor) -> torch.Tensor:
    """x: (B, T, F). Rotate ±10° and scale ±10% in x/y, time warp 0.8–1.2x, drop ~10% frames, swap hands with x-mirror."""
    B, T, F = x.shape
    out = x.clone()
    xyz = out[..., TRIPLES]  # (B, T, N, 3)
    ang = (torch.rand(B, 1, 1) - 0.5) * (2 * np.pi * 10 / 180)
    c, s = torch.cos(ang), torch.sin(ang)
    scale = 1 + (torch.rand(B, 1, 1) - 0.5) * 0.2
    nx = (xyz[..., 0] * c - xyz[..., 1] * s) * scale
    ny = (xyz[..., 0] * s + xyz[..., 1] * c) * scale
    swap = torch.rand(B) < 0.5
    nx = torch.where(swap[:, None, None], -nx, nx)
    out[..., TRIPLES] = torch.stack([nx, ny, xyz[..., 2] * scale], -1)
    out[swap] = out[swap][..., SWAP]
    warp = 0.8 + torch.rand(B) * 0.4
    idx = torch.stack([torch.clamp((torch.arange(T) * w).long(), 0, T - 1) for w in warp])
    idx = torch.where(torch.rand(B, T) > 0.1, idx, torch.roll(idx, 1, dims=1))
    return torch.gather(out, 1, idx[..., None].expand(-1, -1, F))


def fit(X: np.ndarray, y: np.ndarray, Xv: np.ndarray, yv: np.ndarray, n_classes: int, epochs: int = 60,
        device: str | None = None, use_aug: bool = True, patience: int = 10) -> SignNet:
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    model = SignNet(n_classes, X.shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y).long()
    Xvt, yvt = torch.from_numpy(Xv).to(device), torch.from_numpy(yv).long().to(device)
    best, best_state, bad = -1.0, None, 0
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    for ep in range(epochs):
        model.train()
        for b in torch.randperm(len(Xt)).split(64):
            xb = augment(Xt[b]) if use_aug else Xt[b]
            loss = loss_fn(model(xb.to(device)), yt[b].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Xvt).argmax(1) == yvt).float().mean().item() if len(Xvt) else 0.0
        if acc > best:
            best, bad = acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model.cpu().eval()


def topk_acc(model: SignNet, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if not len(X):
        return float("nan"), float("nan")
    with torch.no_grad():
        logits = model(torch.from_numpy(X))
    top = logits.topk(3, dim=1).indices.numpy()
    return float((top[:, 0] == y).mean()), float((top == y[:, None]).any(1).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-signers", default="")
    ap.add_argument("--holdout-signer", default="")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", default=str(DEFAULT_MODEL))
    a = ap.parse_args()
    labels = [s.gloss for s in load_vocab().active]
    X, y = load_dir(LM / "train", labels)
    for signer in filter(None, a.train_signers.split(",")):
        Xo, yo = load_dir(OURS / signer, labels)
        if len(Xo):
            X, y = np.concatenate([X, Xo]), np.concatenate([y, yo])
    Xv, yv = load_dir(LM / "val", labels)
    print(f"train {len(X)}  val {len(Xv)}  classes {len(labels)}")
    model = fit(X, y, Xv, yv, len(labels), a.epochs)
    save_checkpoint(model, labels, a.out)
    Xt, yt = load_dir(LM / "test", labels)
    print("ASL Citizen test top-1 %.3f top-3 %.3f (target top-1 >= 0.80)" % topk_acc(model, Xt, yt))
    if a.holdout_signer:
        Xh, yh = load_dir(OURS / a.holdout_signer, labels)
        print("holdout %s top-1 %.3f top-3 %.3f (target 0.70 / 0.90)" % ((a.holdout_signer,) + topk_acc(model, Xh, yh)))


if __name__ == "__main__":
    main()
