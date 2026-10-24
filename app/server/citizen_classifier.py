"""ASL Citizen's released ST-GCN (Microsoft Research): 2,731 isolated ASL signs from 27 pose + hand keypoints.

Weights: models/asl-citizen/ASL_citizen_stgcn_weights.pt (github.com/microsoft/ASL-citizen-code, release
checkpoints_v1). Label order = sorted glosses of ASL Citizen's splits/train.csv (data/asl_citizen/splits/, fetched by
training.remote_zip). Preprocessing reproduces ST-GCN/asl_citizen_dataset_pose.py exactly, including its quirk of
zero-padding to 128 frames *before* the shoulder normalization. Research use only (ASL Citizen licence).
"""

import csv
import re
from pathlib import Path

import numpy as np
import torch

from .protocol import Frame
from .stgcn.fc import FC
from .stgcn.network import Network
from .stgcn.st_gcn import STGCN

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models" / "asl-citizen" / "ASL_citizen_stgcn_weights.pt"
TRAIN_CSV = ROOT / "data" / "asl_citizen" / "splits" / "train.csv"
MAX_FRAMES = 128
TRAIN_FPS = 30.0  # ASL Citizen webcam videos
# [pose 0-32 | signer's left hand 33-53 | signer's right hand 54-74] -> the 27 graph nodes
KEYPOINTS = [0, 2, 5, 11, 12, 13, 14, 33, 37, 38, 41, 42, 45, 46, 49, 50, 53, 54,
             58, 59, 62, 63, 66, 67, 70, 71, 74]
GRAPH = {"num_nodes": 27, "center": 0,
         "inward_edges": [[2, 0], [1, 0], [0, 3], [0, 4], [3, 5], [4, 6], [5, 7], [6, 17], [7, 8], [7, 9],
                          [9, 10], [7, 11], [11, 12], [7, 13], [13, 14], [7, 15], [15, 16], [17, 18], [17, 19],
                          [19, 20], [17, 21], [21, 22], [17, 23], [23, 24], [17, 25], [25, 26]]}
# ASL Citizen names for a sign whose app gloss is a different English word
ALIASES = {"ME": "I", "BYE": "GOODBYE", "EXCUSE": "EXCUSE-ME"}


def citizen_glosses(path: Path = TRAIN_CSV) -> list[str]:
    with open(path, newline="", encoding="utf-8") as fh:
        return sorted({row[2].strip() for row in list(csv.reader(fh))[1:]})


def to_app_gloss(citizen: str, by_citizen: dict[str, str]) -> str:
    """ASL Citizen gloss -> our gloss when the vocabulary has the sign (variants WANT1/WANT2 merge), else its own
    name without the variant number."""
    base = re.sub(r"\d+$", "", citizen) or citizen
    return by_citizen.get(citizen) or by_citizen.get(base) or by_citizen.get(ALIASES.get(base, "")) or base


def _downsample(frames: np.ndarray, max_frames: int) -> np.ndarray:
    """asl_citizen_dataset_pose.downsample: keep evenly spaced frames, at most max_frames."""
    inc = min(max_frames / len(frames), 1.0)
    out, acc, cur = [], 0.0, 0
    for f in frames:
        acc += inc
        if acc > cur:
            cur += 1
            out.append(f)
    return np.array(out[:max_frames])


def _to_fps(data: np.ndarray, t: np.ndarray, fps: float) -> np.ndarray:
    """Resample (T, 75, 2) frames onto an even fps grid. Pose interpolates; a hand interpolates only between frames
    that both have it, else copies the nearer frame (a missing hand is zeros and must not be blended toward)."""
    grid = np.arange(t[0], t[-1] + 1e-6, 1000.0 / fps)
    i = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 2)
    a = np.clip((grid - t[i]) / np.maximum(t[i + 1] - t[i], 1e-6), 0.0, 1.0)[:, None, None]
    out = data[i] * (1 - a) + data[i + 1] * a
    for lo, hi in ((33, 54), (54, 75)):
        has = data[:, lo:hi].any(axis=(1, 2))
        both = (has[i] & has[i + 1])[:, None, None]
        near = np.where(a[:, 0, 0] < 0.5, i, i + 1)
        out[:, lo:hi] = np.where(both, out[:, lo:hi], data[near, lo:hi])
    return out


def frames_to_input(frames: list[Frame]) -> np.ndarray:
    """App frames -> (2, 128, 27) model input. Each hand goes to the pose wrist it sits on (pose 16 = signer's right),
    which is how MediaPipe Holistic labels hands; the browser's hand labels follow another convention. The browser
    sends ~15 frames/s (MediaPipe's cost) and the model was trained on ~30 fps video, which halved top-1 (benchmark
    half_fps), so sparse input is resampled to TRAIN_FPS by timestamp first."""
    data = np.zeros((len(frames), 75, 2))
    for i, f in enumerate(frames):
        p = np.asarray(f.pose, dtype=np.float64)[:, :2]
        data[i, :33] = p
        for hand in (f.left, f.right):
            if hand is None:
                continue
            h = np.asarray(hand, dtype=np.float64)[:, :2]
            if np.linalg.norm(h[0] - p[16]) < np.linalg.norm(h[0] - p[15]):
                data[i, 54:75] = h  # signer's right hand
            else:
                data[i, 33:54] = h
    t = np.array([f.t for f in frames], dtype=np.float64)
    if len(t) > 1 and np.all(np.diff(t) > 0) and np.median(np.diff(t)) > 1.5 * 1000.0 / TRAIN_FPS:
        data = _to_fps(data, t, TRAIN_FPS)
    if len(data) > MAX_FRAMES:
        data = _downsample(data, MAX_FRAMES)
    if len(data) < MAX_FRAMES:
        data = np.pad(data, ((0, MAX_FRAMES - len(data)), (0, 0), (0, 0)))
    sl, sr = data[:, 11], data[:, 12]
    center = ((sl + sr) / 2).mean(0)
    dist = np.mean(np.sqrt(((sl - sr) ** 2).sum(-1)))
    if dist != 0:
        data = (data - center) / dist
    return np.transpose(data[:, KEYPOINTS], (2, 0, 1)).astype(np.float32)


class CitizenSignPredictor:
    def __init__(self, vocab, model_path: Path | str = DEFAULT_MODEL, train_csv: Path = TRAIN_CSV,
                 restrict_to_vocab: bool = False):
        classes = citizen_glosses(train_csv)
        by_citizen = {s.asl_citizen.upper(): s.gloss for s in vocab.active} | {s.gloss: s.gloss for s in vocab.active}
        self.labels = [to_app_gloss(c, by_citizen) for c in classes]
        self.model = Network(encoder=STGCN(in_channels=2, graph_args=GRAPH, edge_importance_weighting=True),
                             decoder=FC(n_features=256, num_class=len(classes), dropout_ratio=0.05))
        self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        self.model.eval()
        in_vocab = np.array([g in vocab.glosses for g in self.labels])
        # restrict: classes outside our vocabulary can never be predicted (the softmax renormalizes over the rest)
        self.mask = torch.from_numpy(np.where(in_vocab, 0.0, -np.inf).astype(np.float32)) if restrict_to_vocab else None

    def probs(self, frames: list[Frame]) -> np.ndarray:
        x = torch.from_numpy(frames_to_input(frames))[None]
        with torch.no_grad():
            logits = self.model(x)[0]
            if self.mask is not None:
                logits = logits + self.mask
            return torch.softmax(logits, -1).numpy()

    def topk(self, frames: list[Frame], k: int = 3) -> list[tuple[str, float]]:
        merged: dict[str, float] = {}
        for label, p in zip(self.labels, self.probs(frames)):
            merged[label] = merged.get(label, 0.0) + float(p)
        return sorted(merged.items(), key=lambda kv: -kv[1])[:k]

    def recognize(self, seg) -> list[tuple[str, float]]:
        return self.topk(seg.context or seg.frames, 3)
