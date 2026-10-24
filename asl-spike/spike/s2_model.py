"""S2 model: small temporal classifier over normalized landmark sequences (features.py), CPU-trainable.

Conv1d stack -> bidirectional GRU -> attention pooling -> linear. About 1M parameters.
Augmentation (train only): random scale/rotation/shift, time warp (re-sampling a random sub-span),
frame dropout, and dropping a whole hand.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from features import N_FEAT, T


class SignNet(nn.Module):
    def __init__(self, n_cls: int, d: int = 192, drop: float = 0.3):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(N_FEAT * 2, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(0.1))
        self.conv = nn.ModuleList([nn.Sequential(nn.Conv1d(d, d, 5, padding=2, groups=1), nn.BatchNorm1d(d), nn.GELU())
                                   for _ in range(3)])
        self.gru = nn.GRU(d, d // 2, batch_first=True, bidirectional=True)
        self.att = nn.Linear(d, 1)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(d, n_cls))

    def forward(self, x):  # x: (B, T, F)
        v = torch.diff(x, dim=1, prepend=x[:, :1])
        h = self.inp(torch.cat([x, v], -1)).transpose(1, 2)
        for c in self.conv:
            h = h + c(h)
        h, _ = self.gru(h.transpose(1, 2))
        a = torch.softmax(self.att(h).squeeze(-1), 1)
        return self.head((h * a.unsqueeze(-1)).sum(1))


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# feature layout: [lh 42 | rh 42 | pose 16 | face 12 | flags 2]
XY_END = N_FEAT - 2


def augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = x.copy()
    pts = x[:, :XY_END].reshape(T, -1, 2)
    present = np.abs(pts).sum(-1) > 0
    s = rng.uniform(0.85, 1.15)
    a = math.radians(rng.uniform(-12, 12))
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], np.float32)
    sh = rng.uniform(-0.1, 0.1, 2).astype(np.float32)
    pts = (pts @ R.T) * s * np.array([rng.uniform(0.9, 1.1), 1.0], np.float32) + sh
    pts[~present] = 0
    x[:, :XY_END] = pts.reshape(T, -1)
    # time warp: resample a random sub-span back to T frames
    if rng.random() < 0.5:
        lo = rng.uniform(0, 0.15)
        hi = rng.uniform(0.85, 1.0)
        src = np.linspace(lo * (T - 1), hi * (T - 1), T)
        i0 = np.floor(src).astype(int)
        i1 = np.minimum(i0 + 1, T - 1)
        w = (src - i0)[:, None]
        x = x[i0] * (1 - w) + x[i1] * w
    # frame dropout (repeat previous frame)
    for t in range(1, T):
        if rng.random() < 0.08:
            x[t] = x[t - 1]
    # drop the non-dominant hand sometimes (detector misses)
    if rng.random() < 0.1:
        x[:, 0:42] = 0
        x[:, -2] = 0
    return x.astype(np.float32)


def topk_acc(logits: torch.Tensor, y: torch.Tensor, ks=(1, 3, 5)) -> dict:
    out = {}
    top = logits.topk(max(ks), 1).indices
    for k in ks:
        out[f"top{k}"] = float((top[:, :k] == y[:, None]).any(1).float().mean())
    return out
