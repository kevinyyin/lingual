"""Saves recorder takes (landmarks only) under data/ours/. Inputs come from the browser: validate everything."""

import re
from pathlib import Path

from training.landmarks import save_npz

from .protocol import RecordMsg
from .vocab import Vocab

SIGNER = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
GRID = re.compile(r"^_grid/(yn|wh|statement)_(happy|sad|angry|neutral)$")
LETTER = re.compile(r"^_letter/([A-Z])$")
NAME = re.compile(r"^_name/([A-Z]{1,8})$")  # fingerspelled-name takes for the spelling eval


def _next(d: Path, prefix: str = "") -> int:
    taken = {p.stem[len(prefix):] for p in d.glob(f"{prefix}*.npz")}
    n = 0
    while str(n) in taken:
        n += 1
    return n


def save_recording(root: Path, msg: RecordMsg, vocab: Vocab) -> Path:
    if not SIGNER.match(msg.signer):
        raise ValueError(f"bad signer name: {msg.signer!r}")
    if not msg.frames:
        raise ValueError("empty take")
    base = root / msg.signer
    if msg.gloss in vocab.glosses:
        d, prefix = base / msg.gloss, ""
    elif m := GRID.match(msg.gloss):
        d, prefix = base / "_grid", f"{m[1]}_{m[2]}_"
    elif m := LETTER.match(msg.gloss):
        d, prefix = base / "_letters" / m[1], ""
    elif m := NAME.match(msg.gloss):
        d, prefix = base / "_names", f"{m[1]}_"
    else:
        raise ValueError(f"unknown gloss: {msg.gloss!r}")
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{prefix}{_next(d, prefix)}.npz"
    save_npz(msg.frames, path)
    return path
