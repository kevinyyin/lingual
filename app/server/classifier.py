"""Isolated-sign classifier over normalized landmark sequences."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .features import FEAT_DIM, sequence_features
from .protocol import Frame

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "signs.pt"


class SignNet(nn.Module):
    """2-layer bidirectional GRU over T frames, mean+max pooled."""

    def __init__(self, n_classes: int, feat_dim: int = FEAT_DIM, T: int = 32, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.T = T
        self.inp = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * hidden, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, F) -> (B, C) logits
        h, _ = self.gru(self.inp(x))
        return self.head(torch.cat([h.mean(1), h.amax(1)], dim=-1))


def save_checkpoint(model: SignNet, labels: list[str], path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "labels": labels, "feat_dim": FEAT_DIM, "T": model.T}, path)


class SignPredictor:
    def __init__(self, model_path: Path | str = DEFAULT_MODEL):
        ck = torch.load(model_path, map_location="cpu", weights_only=True)
        self.labels: list[str] = ck["labels"]
        self.T = ck["T"]
        self.model = SignNet(len(self.labels), ck["feat_dim"], self.T)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()

    def topk(self, frames: list[Frame], k: int = 3) -> list[tuple[str, float]]:
        x = torch.from_numpy(sequence_features(frames, self.T)[None])
        with torch.no_grad():
            p = torch.softmax(self.model(x), dim=-1)[0].numpy()
        idx = np.argsort(-p)[:k]
        return [(self.labels[i], float(p[i])) for i in idx]

    def recognize(self, seg) -> list[tuple[str, float]]:
        return self.topk(seg.frames, 3)
