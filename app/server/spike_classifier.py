"""The asl-spike's trained isolated-sign model (143 café signs) running on the app's live frames.

Weights: models/asl-sign-models/weights/isolated/model.pt (teammate bundle; ASL Citizen-derived, team-internal only).
SignNet and normalize() are ported unchanged from the bundle's code/spike/s2_model.py and features.py, so live input
gets exactly the preprocessing the weights were trained with. Input per frame: both hands (21 x,y), 8 pose points,
6 face-mesh points (Frame.face, sent by camera.ts).
"""

import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .protocol import Frame

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "asl-sign-models" / "weights" / "isolated" / "model.pt"

T = 32
POSE_IDX = [0, 11, 12, 13, 14, 15, 16, 23]  # nose, shoulders, elbows, wrists, left hip
POSE_MIRROR = [0, 2, 1, 4, 3, 6, 5, 7]
N_FEAT = 2 * 21 * 2 + len(POSE_IDX) * 2 + 6 * 2 + 2  # hands, pose, face, 2 hand-present flags
# spike class -> our gloss where the names differ (IX-1P: index finger to chest, "I")
RENAME = {"BYE": "GOODBYE", "IX-1P": "I"}


class SignNet(nn.Module):
    """Conv1d stack -> bidirectional GRU -> attention pooling -> linear (spike s2_model.py)."""

    def __init__(self, n_cls: int, d: int = 192, drop: float = 0.3):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(N_FEAT * 2, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(0.1))
        self.conv = nn.ModuleList([nn.Sequential(nn.Conv1d(d, d, 5, padding=2, groups=1), nn.BatchNorm1d(d), nn.GELU())
                                   for _ in range(3)])
        self.gru = nn.GRU(d, d // 2, batch_first=True, bidirectional=True)
        self.att = nn.Linear(d, 1)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(d, n_cls))

    def forward(self, x):  # (B, T, F) -> (B, C) logits
        v = torch.diff(x, dim=1, prepend=x[:, :1])
        h = self.inp(torch.cat([x, v], -1)).transpose(1, 2)
        for c in self.conv:
            h = h + c(h)
        h, _ = self.gru(h.transpose(1, 2))
        a = torch.softmax(self.att(h).squeeze(-1), 1)
        return self.head((h * a.unsqueeze(-1)).sum(1))


def _motion(h: np.ndarray) -> float:
    with warnings.catch_warnings():  # frames without this hand are all-NaN: "mean of empty slice" on every one-handed sign
        warnings.simplefilter("ignore", RuntimeWarning)
        c = np.nanmean(h, axis=1)
    v = np.diff(c, axis=0)
    ok = ~np.isnan(v).any(1)
    return float(np.abs(v[ok]).sum()) + 0.05 * float((~np.isnan(c).any(1)).sum())


def normalize(lh, rh, pose, face, t: int = T) -> np.ndarray:
    """Spike features.normalize: dominant hand into the right slot (mirroring x), shoulder-centred and scaled, trimmed
    to hand-visible frames, resampled to t frames."""
    lh, rh, pose, face = (a.astype(np.float32).copy() for a in (lh, rh, pose, face))
    if _motion(lh) > _motion(rh):
        lh, rh = rh, lh
        for a in (lh, rh, pose, face):
            a[..., 0] = 1.0 - a[..., 0]
        pose = pose[:, POSE_MIRROR]
    sh = pose[:, 1:3]
    mid = np.nanmedian(sh.mean(1), axis=0)
    width = np.nanmedian(np.linalg.norm(sh[:, 0] - sh[:, 1], axis=-1))
    if not np.isfinite(width) or width < 1e-3:
        mid = np.nanmedian(np.nanmean(np.concatenate([lh, rh], 1), 1), 0)
        width = 0.3
    if not np.all(np.isfinite(mid)):
        mid = np.array([0.5, 0.5], np.float32)
    lh, rh, pose, face = ((a - mid) / width for a in (lh, rh, pose, face))
    present = ~(np.isnan(lh).all((1, 2)) & np.isnan(rh).all((1, 2)))
    idx = np.where(present)[0]
    lo, hi = (max(0, idx[0] - 3), min(len(present), idx[-1] + 4)) if len(idx) else (0, len(present))
    seq = [a[lo:hi] for a in (lh, rh, pose, face)]
    F = hi - lo
    flags = np.stack([~np.isnan(seq[0]).all((1, 2)), ~np.isnan(seq[1]).all((1, 2))], 1).astype(np.float32)
    x = np.concatenate([seq[0].reshape(F, -1), seq[1].reshape(F, -1), seq[2].reshape(F, -1), seq[3].reshape(F, -1),
                        flags], 1)
    x = np.nan_to_num(x, nan=0.0)
    if F == 0:
        return np.zeros((t, N_FEAT), np.float32)
    src = np.linspace(0, F - 1, t)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, F - 1)
    w = (src - i0)[:, None]
    return (x[i0] * (1 - w) + x[i1] * w).astype(np.float32)


def frames_to_arrays(frames: list[Frame]) -> tuple[np.ndarray, ...]:
    """App frames -> spike (lh, rh, pose8, face6), each (F, n, 2), NaN where missing. Holistic's lh/rh are the signer's
    own hands; the browser's hand labels follow another convention, so each hand goes to the pose wrist it sits on."""
    F = len(frames)
    lh, rh = np.full((F, 21, 2), np.nan, np.float32), np.full((F, 21, 2), np.nan, np.float32)
    pose = np.full((F, len(POSE_IDX), 2), np.nan, np.float32)
    face = np.full((F, 6, 2), np.nan, np.float32)
    for i, f in enumerate(frames):
        p = np.asarray(f.pose, np.float32)[:, :2]
        pose[i] = p[POSE_IDX]
        if f.face is not None and len(f.face) == 6:
            face[i] = np.asarray(f.face, np.float32)[:, :2]
        for hand in (f.left, f.right):
            if hand is None:
                continue
            h = np.asarray(hand, np.float32)[:, :2]
            right = np.linalg.norm(h[0] - p[16]) < np.linalg.norm(h[0] - p[15])  # pose 16 = signer's right wrist
            (rh if right else lh)[i] = h
    return lh, rh, pose, face


class SpikeSignPredictor:
    def __init__(self, model_path: Path | str = DEFAULT_MODEL):
        ck = torch.load(model_path, map_location="cpu", weights_only=True)
        self.labels = [RENAME.get(c, c) for c in ck["classes"]]
        self.model = SignNet(len(self.labels))
        self.model.load_state_dict(ck["state"])
        self.model.eval()

    def topk(self, frames: list[Frame], k: int = 3) -> list[tuple[str, float]]:
        x = torch.from_numpy(normalize(*frames_to_arrays(frames))[None])
        with torch.no_grad():
            p = torch.softmax(self.model(x), dim=-1)[0].numpy()
        merged: dict[str, float] = {}
        for label, prob in zip(self.labels, p):
            merged[label] = merged.get(label, 0.0) + float(prob)
        return sorted(merged.items(), key=lambda kv: -kv[1])[:k]

    def recognize(self, seg) -> list[tuple[str, float]]:
        return self.topk(seg.context or seg.frames, 3)
