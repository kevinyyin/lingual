"""Train the fingerspelling letter model from our recorded letters, and build the avatar's letter clips.

Data: data/ours/{signer}/_letters/{A..Z}/*.npz (recorder: ?mode=record&letters=1&signer=NAME).
Run: uv run python -m training.train_fingerspell --train-signers kev,ana --holdout-signer sam [--clips-from kev]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from server.fingerspell import DEFAULT_MODEL, LETTERS, LetterNet, save_letter_model, windows
from server.protocol import Frame
from training.landmarks import load_npz

ROOT = Path(__file__).resolve().parent.parent
OURS = ROOT / "data" / "ours"
LETTER_CLIPS = ROOT / "web" / "public" / "letters"
KEEP = (0.2, 0.8)  # middle of each take: skip moving into and out of the handshape


def letter_dataset(root: Path, signers: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for signer in signers:
        for i, ch in enumerate(LETTERS):
            for p in sorted((root / signer / "_letters" / ch).glob("*.npz")):
                frames = load_npz(p)
                frames = frames[int(len(frames) * KEEP[0]):int(len(frames) * KEEP[1])] or frames
                if any(f.left or f.right for f in frames):
                    w = windows(frames)
                    X.append(w)
                    y += [i] * len(w)
    if not X:
        return np.zeros((0, 0), np.float32), np.zeros(0, int)
    return np.concatenate(X).astype(np.float32), np.array(y)


def fit(X: np.ndarray, y: np.ndarray, epochs: int = 80) -> LetterNet:
    model = LetterNet()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y).long()
    for _ in range(epochs):
        model.train()
        for b in torch.randperm(len(Xt)).split(128):
            xb = Xt[b] * (1 + (torch.rand(len(b), 1) - 0.5) * 0.2)  # small scale jitter
            loss = nn.functional.cross_entropy(model(xb), yt[b], label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


def letter_clip(frames: list[Frame]) -> str:
    """Avatar clip from a recorded take: recordings have image landmarks only, so the world pose is approximated
    from them (hip-centred, 4:3 camera, scaled to ~0.36 m shoulders). Only the arm position and handshape matter here."""
    frames = frames[int(len(frames) * KEEP[0]):int(len(frames) * KEEP[1])] or frames
    out = []
    for f in frames:
        p = np.asarray(f.pose, dtype=np.float64)[:, :3]
        p[:, 0] *= 4 / 3
        hip = (p[23] + p[24]) / 2
        width = max(np.linalg.norm((p[11] - p[12])[:2]), 1e-3)
        world = (p - hip) * (0.36 / width)
        out.append({"pose": world.tolist(), "pose2d": [q[:2] for q in f.pose], "left": f.left, "right": f.right})
    return json.dumps({"fps": 30.0, "frames": out})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-signers", required=True)
    ap.add_argument("--holdout-signer", default="")
    ap.add_argument("--clips-from", default="", help="signer whose takes become web/public/letters/*.json")
    a = ap.parse_args()
    X, y = letter_dataset(OURS, a.train_signers.split(","))
    print(f"{len(X)} training windows")
    model = fit(X, y)
    save_letter_model(model, DEFAULT_MODEL)
    if a.holdout_signer:
        Xh, yh = letter_dataset(OURS, [a.holdout_signer])
        if len(Xh):
            with torch.no_grad():
                pred = model(torch.from_numpy(Xh)).argmax(1).numpy()
            acc = [(pred[yh == i] == i).mean() for i in range(26) if (yh == i).any()]
            print(f"holdout per-letter accuracy: mean {np.mean(acc):.3f} (target >= 0.85)")
    if a.clips_from:
        LETTER_CLIPS.mkdir(parents=True, exist_ok=True)
        for ch in LETTERS:
            takes = sorted((OURS / a.clips_from / "_letters" / ch).glob("*.npz"))
            if takes:
                best = max((load_npz(p) for p in takes), key=lambda fs: sum(bool(f.left or f.right) for f in fs))
                (LETTER_CLIPS / f"{ch}.json").write_text(letter_clip(best))
        print(f"letter clips written to {LETTER_CLIPS}")


if __name__ == "__main__":
    main()
