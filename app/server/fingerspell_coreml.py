"""Fingerspelling reader: a pretrained CTC model over MediaPipe Holistic landmarks, run by Core ML on the Mac
(huggingface.co/Skunk0/openglasses-fingerspelling-ctc, Apache-2.0; trained on Google's Kaggle ASL Fingerspelling
corpus, 100+ signers, CC-BY 4.0). Its input format is undocumented beyond shapes; tried on Kaggle sequences, the right
one (points' x, y, z side by side in Holistic order, each axis standardised over the window) gave its published CER.

Kaggle sequences fed as the app gets them (15 frames/s resampled to 30, face mesh to 3 decimals): character error rate
0.18 on 150 phrases, 0.16 on name-like ones (full-rate Holistic 0.15; our own CTC reader, fingerspell_ctc.py, 0.43).
Those sequences may be in its training set, so new signers on a laptop will read worse. The face matters: the window's
standardisation is mostly the face's, and without the mesh (Frame.mesh) CER was 0.39.

Install: uv run --with huggingface_hub python -c "from huggingface_hub import snapshot_download;
snapshot_download('Skunk0/openglasses-fingerspelling-ctc', local_dir='models/fingerspell_coreml',
allow_patterns=['*.mlpackage/*', 'vocab.txt', 'LICENSE'])"
"""

import math
from pathlib import Path

import numpy as np

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "fingerspell_coreml"
T_MAX = 768  # the converted model's fixed window: 25 s at 30 frames/s
FPS = 30.0  # the corpus's rate; the browser's ~15 frames/s raised CER from 0.15 to 0.21 before resampling


def _slot(f) -> tuple[list | None, list | None]:
    """(left, right) hands by the signer's own side: each hand goes to the nearer pose wrist (15 left, 16 right), as in
    Holistic; the hand tracker's labels disagree on ~4% of frames."""
    hands = [h for h in (f.left, f.right) if h is not None]
    if not hands:
        return None, None
    wl, wr = np.asarray(f.pose[15][:2]), np.asarray(f.pose[16][:2])
    d = [(np.linalg.norm(np.asarray(h[0][:2]) - wl), np.linalg.norm(np.asarray(h[0][:2]) - wr)) for h in hands]
    if len(hands) == 1:
        return (hands[0], None) if d[0][0] < d[0][1] else (None, hands[0])
    return (hands[0], hands[1]) if d[0][0] + d[1][1] <= d[0][1] + d[1][0] else (hands[1], hands[0])


def holistic(frames) -> np.ndarray:
    """(T, 543, 3) in Holistic order: face 0-467, left hand 468-488, pose 489-521, right hand 522-542; NaN if missing."""
    x = np.full((len(frames), 543, 3), np.nan, np.float32)
    for i, f in enumerate(frames):
        left, right = _slot(f)
        if f.mesh is not None:
            x[i, :468] = f.mesh
        if left is not None:
            x[i, 468:489] = left
        x[i, 489:522] = [p[:3] for p in f.pose]
        if right is not None:
            x[i, 522:543] = right
    return x


def resample(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Onto an even FPS grid by time: linear between frames; where a point is missing on one side, the nearer frame."""
    if len(x) < 2:
        return x
    grid = np.arange(t[0], t[-1] + 1e-6, 1000.0 / FPS)
    i = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 2)
    w = ((grid - t[i]) / np.maximum(t[i + 1] - t[i], 1e-6)).clip(0, 1)[:, None, None]
    a, b = x[i], x[i + 1]
    out = (1 - w) * a + w * b
    return np.where(np.isnan(out), np.where(w < 0.5, a, b), out).astype(np.float32)


def standardize(x: np.ndarray) -> np.ndarray:
    """Each axis to mean 0, std 1 over every present point of the window; missing points 0."""
    mean = np.nanmean(x, (0, 1), keepdims=True)
    std = np.nanstd(x, (0, 1), keepdims=True)
    return np.nan_to_num((x - mean) / (std + 1e-6), nan=0.0)


def decode(logits: np.ndarray, vocab: list[str]) -> tuple[str, float]:
    """CTC greedy over letters only (names and words: the corpus's digits and punctuation never win) -> (text, mean
    probability of the letters emitted)."""
    keep = np.full(logits.shape[1], -np.inf, np.float32)
    keep[[0] + [k for k, c in enumerate(vocab) if c.isalpha()]] = 0.0
    z = logits + keep
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    out, conf, prev = [], [], 0
    for t, k in enumerate(p.argmax(1)):
        if k != prev and k != 0:
            out.append(vocab[k])
            conf.append(float(p[t, k]))
        prev = k
    return "".join(out), float(np.mean(conf)) if conf else 0.0


class CoreMLSpellReader:
    def __init__(self, model_dir: Path | str = DEFAULT_MODEL):
        import coremltools as ct

        model_dir = Path(model_dir)
        self.model = ct.models.MLModel(str(model_dir / "Fingerspelling2P.mlpackage"))
        self.vocab = (model_dir / "vocab.txt").read_text().split("\n")[:60]  # <blank> + 59 symbols; 2 aux outputs unused

    def read(self, frames) -> tuple[str, float]:
        x = resample(holistic(frames), np.array([f.t for f in frames], np.float64))[:T_MAX]
        T = len(x)
        if T < 2:
            return "", 0.0
        feat = np.zeros((1, T_MAX, 543 * 3), np.float32)
        feat[0, :T] = standardize(x).reshape(T, -1)
        mask = np.zeros((1, T_MAX), np.float32)
        mask[0, :T] = 1.0
        logits = np.asarray(self.model.predict({"features": feat, "mask": mask})["Identity"], np.float32)[0]
        return decode(logits[: math.ceil(T / 2)], self.vocab)  # the model halves time

    def recognize(self, seg) -> list[tuple[str, float]]:
        """App segment -> [(WORD, confidence)]: letters only, as the other spelling readers."""
        text, conf = self.read(seg.frames)
        word = text.upper()
        return [(word, conf)] if word else []
