"""Fingerspelling recognition on real signers: Google's Kaggle ASL Fingerspelling landmarks (100+ Deaf signers).

Runs the MIT-licensed static-letter classifier (huggingface.co/AmimulBmeIU/asl-alphabet-classifier; 21 hand landmarks,
A-Y without the moving J and Z) frame by frame on the signing hand and decodes letters with the app's own
server.fingerspell.decode, then scores character error rate against the phrase's letters. The input normalization
is undocumented, so variants are chosen on one half of the sequences and reported on the other half.

Needs: data/kaggle_fs/train.csv + one train_landmarks/*.parquet (kaggle competitions download -c asl-fingerspelling),
models/letters-mit/letters_mit.npz. Run: uv run --with pyarrow python -m training.eval_fingerspell
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from server.fingerspell import LETTERS, decode

ROOT = Path(__file__).resolve().parent.parent
FS = ROOT / "data" / "kaggle_fs"
MIT = ROOT / "models" / "letters-mit"


def mlp(weights: dict, x: np.ndarray) -> np.ndarray:
    h = np.maximum(x @ weights["w0"] + weights["b0"], 0)
    h = np.maximum(h @ weights["w1"] + weights["b1"], 0)
    z = h @ weights["w2"] + weights["b2"]
    z = np.exp(z - z.max(-1, keepdims=True))
    return z / z.sum(-1, keepdims=True)


def normalize(hand: np.ndarray, how: str) -> np.ndarray:
    """hand: (n, 21, 3) -> (n, 63)."""
    if how == "raw":
        h = hand
    else:
        h = hand - hand[:, :1]
        if how == "wrist_scaled":
            h = h / np.maximum(np.abs(h).reshape(len(h), -1).max(1), 1e-6)[:, None, None]
    return h.reshape(len(h), -1).astype(np.float32)


def to26(p24: np.ndarray, labels: list[str]) -> np.ndarray:
    out = np.zeros((len(p24), 26), np.float32)
    for j, l in enumerate(labels):
        out[:, LETTERS.index(l)] = p24[:, j]
    return out


def cer(ref: str, hyp: str) -> float:
    d = np.arange(len(hyp) + 1)
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1] / max(len(ref), 1)


def sequences(parquet: Path, csv: Path) -> list[tuple[str, np.ndarray]]:
    """(phrase letters, signing hand (n, 21, 3)) per sequence: the hand present in more frames, mirrored if left."""
    import pandas as pd

    meta = pd.read_csv(csv)
    meta = meta[meta.file_id == int(parquet.stem)].set_index("sequence_id")
    cols = [f"{a}_{s}_hand_{i}" for s in ("right", "left") for a in "xyz" for i in range(21)]
    df = pd.read_parquet(parquet, columns=cols)
    out = []
    for seq, g in df.groupby(level=0):
        if seq not in meta.index:
            continue
        phrase = str(meta.loc[seq, "phrase"])
        letters = re.sub(r"[^A-Z]", "", phrase.upper())
        if len(letters) < 0.8 * len(phrase.replace(" ", "")) or len(letters) < 2:
            continue  # phone numbers, URLs: this model reads letters only
        arr = {s: g[[f"{a}_{s}_hand_{i}" for i in range(21) for a in "xyz"]].to_numpy().reshape(len(g), 21, 3) for s in ("right", "left")}
        side = max(arr, key=lambda s: (~np.isnan(arr[s][:, 0, 0])).sum())
        hand = arr[side][~np.isnan(arr[side][:, 0, 0])]
        if len(hand) < 4:
            continue
        if side == "left":
            hand = hand.copy()
            hand[..., 0] = 1.0 - hand[..., 0]  # one-handed model: mirror a left hand into a right hand
        out.append((letters, hand))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, default=next(iter(sorted((FS / "train_landmarks").glob("*.parquet"))), None))
    ap.add_argument("--max", type=int, default=600)
    a = ap.parse_args()
    weights = dict(np.load(MIT / "letters_mit.npz"))
    labels = json.loads((MIT / "labels.json").read_text(encoding="utf-8"))
    seqs = sequences(a.parquet, FS / "train.csv")[: a.max]
    tune, test = seqs[::2], seqs[1::2]

    def run(group, how):
        rows = []
        for letters, hand in group:
            p = to26(mlp(weights, normalize(hand, how)), labels)
            words = decode(p)
            rows.append((letters, words[0][0] if words else ""))
        return rows

    scores = {how: np.mean([cer(r, h) for r, h in run(tune, how)]) for how in ("raw", "wrist", "wrist_scaled")}
    best = min(scores, key=scores.get)
    rows = run(test, best)
    c = [cer(r, h) for r, h in rows]
    lines = [f"# Fingerspelling: MIT static-letter model on Kaggle ASL Fingerspelling ({a.parquet.name})", "",
             f"normalization chosen on {len(tune)} tuning sequences: {best} (CER by variant: "
             + ", ".join(f"{k} {v:.3f}" for k, v in scores.items()) + ")",
             f"held-out {len(test)} sequences: mean CER {np.mean(c):.3f}, median {np.median(c):.3f}, "
             f"exactly right {np.mean([x == 0 for x in c]):.1%}, CER <= 0.25 {np.mean([x <= 0.25 for x in c]):.1%}", "",
             "examples (truth -> read):"] + [f"- {r} -> {h or '(nothing)'}" for r, h in rows[:15]]
    (ROOT / "data" / "eval").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "eval" / "fingerspell_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
