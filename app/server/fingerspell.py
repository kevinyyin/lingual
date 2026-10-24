"""Fingerspelling: handshape classifier over short windows of one hand, decoded into a word plus alternatives."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .protocol import Frame

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
WINDOW = 8  # frames per window: enough to see the motion in J and Z
STRIDE = 2
MIN_P = 0.5  # windows less confident than this are treated as transitions
MIN_ALT_P = 0.05
DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "letters.pt"


def hand_features(hand) -> np.ndarray:
    h = np.asarray(hand, dtype=np.float32)
    rel = h - h[0]
    size = max(float(np.linalg.norm(rel[9][:2])), 1e-3)  # wrist -> middle-finger knuckle
    return (rel / size).ravel()


def windows(frames: list[Frame]) -> np.ndarray:
    """(n_windows, WINDOW * 63) from the segment's signing hand (the one present most often)."""
    side = "right" if sum(f.right is not None for f in frames) >= sum(f.left is not None for f in frames) else "left"
    feats = [hand_features(getattr(f, side)) if getattr(f, side) is not None else np.zeros(63, np.float32) for f in frames]
    while len(feats) < WINDOW:
        feats.append(feats[-1] if feats else np.zeros(63, np.float32))
    x = np.stack(feats)
    return np.stack([x[i:i + WINDOW].ravel() for i in range(0, len(x) - WINDOW + 1, STRIDE)])


class LetterNet(nn.Module):
    def __init__(self, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(WINDOW * 63, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, len(LETTERS)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def save_letter_model(model: LetterNet, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, path)


def decode(p: np.ndarray) -> list[tuple[str, float]]:
    """p: (n_windows, 26) letter probabilities -> [(word, confidence), ...], best first.

    Repeats collapse into one letter; a low-confidence dip between two runs of the same letter makes a double letter.
    Alternatives swap in the second-best letter at the least confident positions.
    """
    runs: list[list[int]] = []  # window indices per letter occurrence
    prev = None
    for i, row in enumerate(p):
        best = int(row.argmax())
        if row[best] < MIN_P:
            prev = None
            continue
        if best == prev:
            runs[-1].append(i)
        else:
            runs.append([i])
        prev = best
    if not runs:
        return []
    means = [p[r].mean(0) for r in runs]
    letters = [int(m.argmax()) for m in means]
    conf = [float(m[l]) for m, l in zip(means, letters)]
    word = "".join(LETTERS[l] for l in letters)
    out = [(word, min(conf))]
    for pos in np.argsort(conf)[:2]:
        second = int(np.argsort(-means[pos])[1])
        alt_p = float(means[pos][second])
        alt = word[:pos] + LETTERS[second] + word[pos + 1:]
        if alt_p >= MIN_ALT_P and alt not in dict(out):
            out.append((alt, alt_p))
    return sorted(out, key=lambda w: -w[1])


class SpellReader:
    def __init__(self, model_path: Path | str = DEFAULT_MODEL):
        self.model = LetterNet()
        self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True)["state_dict"])
        self.model.eval()

    def recognize(self, seg) -> list[tuple[str, float]]:
        with torch.no_grad():
            p = torch.softmax(self.model(torch.from_numpy(windows(seg.frames))), dim=-1).numpy()
        return decode(p)
