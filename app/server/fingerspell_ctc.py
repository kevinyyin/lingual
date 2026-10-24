"""Continuous fingerspelling reader (CTC), trained on Google's Kaggle ASL Fingerspelling landmarks (training/
train_fingerspell_ctc.py). Shared by training and runtime, like features.py, so both see identical features.

Per frame: both hands' 21 landmarks, wrist-centred and scaled by hand size; the signing hand (the one present in
more frames) is mirrored into the right-hand slot so left-handed spellers look the same. Model: a small Conv1D +
Transformer encoder with a CTC head over the competition's characters; the app keeps letters only.
"""

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "fingerspell_ctc.pt"
FEAT = 2 * 21 * 3 + 2  # two hands + presence flags
MAX_T = 256


def hand_feat(h: np.ndarray) -> np.ndarray:
    """(T, 21, 3) with NaN rows for missing frames -> wrist-centred, hand-size-scaled (T, 63); NaN -> 0."""
    rel = h - h[:, :1]
    size = np.linalg.norm(rel[:, 9, :2], axis=-1, keepdims=True)[:, :, None]
    out = rel / np.maximum(np.nan_to_num(size, nan=1.0), 1e-3)
    return np.nan_to_num(out.reshape(len(h), -1), nan=0.0)


def features(right: np.ndarray, left: np.ndarray) -> np.ndarray:
    """right/left: (T, 21, 3) image landmarks with NaN where a hand is missing -> (T, FEAT)."""
    r_ok, l_ok = ~np.isnan(right[:, 0, 0]), ~np.isnan(left[:, 0, 0])
    if l_ok.sum() > r_ok.sum():  # signing hand into the right-hand slot, mirrored
        right, left, r_ok, l_ok = left.copy(), right.copy(), l_ok, r_ok
        right[..., 0] *= -1
        left[..., 0] *= -1
    x = np.concatenate([hand_feat(right), hand_feat(left), r_ok[:, None], l_ok[:, None]], 1).astype(np.float32)
    keep = r_ok | l_ok
    return x[keep] if keep.any() else x[:0]


class SpellNet(nn.Module):
    def __init__(self, n_out: int, d: int = 192, layers: int = 3, heads: int = 4, drop: float = 0.15):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(FEAT, d), nn.LayerNorm(d), nn.GELU())
        self.conv = nn.ModuleList([nn.Sequential(nn.Conv1d(d, d, 5, padding=2, groups=4), nn.BatchNorm1d(d), nn.GELU())
                                   for _ in range(3)])
        self.pos = nn.Parameter(torch.zeros(1, MAX_T, d))
        enc = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, batch_first=True, activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)  # unused with norm_first
        self.out = nn.Linear(d, n_out + 1)  # + CTC blank (index 0)

    def forward(self, x: torch.Tensor, pad: torch.Tensor | None = None) -> torch.Tensor:  # (B, T, F) -> (B, T, C)
        h = self.inp(x).transpose(1, 2)
        for c in self.conv:
            h = h + c(h)
        h = h.transpose(1, 2) + self.pos[:, : h.shape[2]]
        return self.out(self.enc(h, src_key_padding_mask=pad))


def greedy(logp: np.ndarray, chars: list[str]) -> tuple[str, float]:
    """CTC greedy decode -> (text, mean confidence of the emitted characters)."""
    best = logp.argmax(-1)
    out, conf, prev = [], [], 0
    for t, k in enumerate(best):
        if k != prev and k != 0:
            out.append(chars[k - 1])
            conf.append(math.exp(logp[t, k]))
        prev = k
    return "".join(out), float(np.mean(conf)) if conf else 0.0


class CtcSpellReader:
    def __init__(self, model_path: Path | str = DEFAULT_MODEL):
        ck = torch.load(model_path, map_location="cpu", weights_only=True)
        self.chars: list[str] = ck["chars"]
        self.model = SpellNet(len(self.chars))
        self.model.load_state_dict(ck["state"])
        self.model.eval()

    def read(self, right: np.ndarray, left: np.ndarray) -> tuple[str, float]:
        x = features(right, left)
        if len(x) < 2:
            return "", 0.0
        if len(x) > MAX_T:
            x = x[np.linspace(0, len(x) - 1, MAX_T).astype(int)]
        with torch.no_grad():
            logp = torch.log_softmax(self.model(torch.from_numpy(x)[None]), -1)[0].numpy()
        return greedy(logp, self.chars)

    def recognize(self, seg) -> list[tuple[str, float]]:
        """App segment -> [(WORD, confidence)]: letters only (names and words; digits are not part of the turn)."""
        def track(side):
            return np.array([np.asarray(getattr(f, side), np.float32) if getattr(f, side) is not None
                             else np.full((21, 3), np.nan, np.float32) for f in seg.frames])
        text, conf = self.read(track("right"), track("left"))
        word = "".join(c for c in text.upper() if "A" <= c <= "Z")
        return [(word, conf)] if word else []
