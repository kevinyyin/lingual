"""Avatar fingerspelling clips (web/public/letters/A.json ... Z.json) from real Deaf signers' handshapes.

Google's Kaggle ASL Fingerspelling landmarks show thousands of letters in context. The CTC reader trained on them
(server/fingerspell_ctc.py) fires on one frame per letter; on sequences it reads exactly right, that frame is the
letter. Per letter the most typical of those real hands (the medoid, so never an average blur of hands) becomes the
clip's handshape, held for a beat; J and Z keep their motion. The arm comes from a sign clip frame with the right hand
raised in front of the shoulder, where fingerspelling is done.

Kaggle stores image coordinates (x by width, y by height); each sequence's height/width ratio is estimated from the
hand itself (bone lengths stay constant as the hand turns only under the right ratio). Non-commercial research use.

Run (after training.train_fingerspell_ctc; parquets in data/kaggle_fs):
  uv run --with pyarrow --with pandas python -m training.letters_from_kaggle
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from server.fingerspell_ctc import DEFAULT_MODEL, MAX_T, SpellNet, features
from training.hand_cleanup import fix_depth, fix_knuckles

ROOT = Path(__file__).resolve().parent.parent
FS = ROOT / "data" / "kaggle_fs"
OUT = ROOT / "web" / "public" / "letters"
SIGNS = ROOT / "web" / "public" / "signs"
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)]
PALM_M = 0.09  # wrist to middle knuckle, metres (adult hand)
HOLD = 7  # frames a static letter is held at 30 fps (~4 letters/s, a relaxed spelling pace)
MOVING = {"J": (-9, 3), "Z": (-12, 3)}  # frames around the CTC spike kept for letters drawn in the air
FPS = 30.0
MEDOID_MAX = 400  # samples compared per letter


def aspect(hands: np.ndarray) -> float:
    """Height/width ratio under which this sequence's bone lengths vary least across frames (the hand turns)."""
    best, best_cv = 1.0, np.inf
    for s in np.linspace(0.4, 2.5, 43):
        h = hands * np.array([1.0, s, 1.0])
        lengths = np.stack([np.linalg.norm(h[:, a] - h[:, b], axis=-1) for a, b in BONES], 1)
        lengths = lengths / lengths.mean(0, keepdims=True)
        cv = float(np.nanstd(lengths, 0).mean())
        if cv < best_cv:
            best, best_cv = s, cv
    return best


def normalize(hand: np.ndarray) -> np.ndarray:
    """(21, 3) metric-ratio hand -> depth- and knuckle-corrected, wrist at origin, palm length PALM_M."""
    rel = fix_knuckles(fix_depth(hand)) - hand[0]
    return rel * (PALM_M / max(np.linalg.norm(rel[9]), 1e-6))


def spikes(logp: np.ndarray) -> list[tuple[int, int]]:
    """CTC greedy path -> [(frame, class)] at the first frame of each emitted character."""
    best, out, prev = logp.argmax(-1), [], 0
    for t, k in enumerate(best):
        if k != prev and k != 0:
            out.append((t, int(k)))
        prev = k
    return out


def collect(model, chars, meta, parquet: Path, samples: dict, motion: dict) -> tuple[int, int]:
    import pandas as pd

    cols = [f"{a}_{s}_hand_{i}" for s in ("right", "left") for i in range(21) for a in "xyz"]
    df = pd.read_parquet(parquet, columns=cols)
    phrases = meta[meta.file_id == int(parquet.stem)].set_index("sequence_id")["phrase"]
    used = total = 0
    for seq, g in df.groupby(level=0):
        if seq not in phrases.index:
            continue
        total += 1
        a = g.to_numpy(np.float32).reshape(len(g), 2, 21, 3)
        right, left = a[:, 0], a[:, 1]
        r_ok, l_ok = ~np.isnan(right[:, 0, 0]), ~np.isnan(left[:, 0, 0])
        lefty = l_ok.sum() > r_ok.sum()
        hand, ok = (left.copy(), l_ok) if lefty else (right.copy(), r_ok)
        if lefty:
            hand[..., 0] = 1.0 - hand[..., 0]  # a left-handed speller mirrored into a right hand, like features()
        x = features(right, left)
        frames = np.flatnonzero(r_ok | l_ok)  # features() keeps only frames with a hand
        if len(x) < 4 or len(x) > MAX_T:
            continue
        with torch.no_grad():
            logp = torch.log_softmax(model(torch.from_numpy(x)[None]), -1)[0].numpy()
        sp = spikes(logp)
        if "".join(chars[k - 1] for _, k in sp).upper() != str(phrases.loc[seq]).upper():
            continue  # only sequences read exactly right: then spike i is phrase letter i
        good = hand[ok]
        if len(good) < 8:
            continue
        scale = np.array([1.0, aspect(good), 1.0])
        used += 1
        for t, k in sp:
            c = chars[k - 1].upper()
            if not re.fullmatch(r"[A-Z]", c):
                continue
            f = frames[t]
            if c in MOVING:
                lo, hi = MOVING[c]
                win = [hand[i] for i in range(f + lo, f + hi + 1) if 0 <= i < len(hand) and ok[i]]
                if len(win) >= (hi - lo) * 0.7:
                    motion[c].append([normalize(h * scale) for h in win])
            if ok[f]:
                samples[c].append((normalize(hand[f] * scale), float(np.exp(logp[t, k]))))
    return used, total


def medoid(xs: list[np.ndarray]) -> int:
    """Index of the sample closest to all the others (a sample of up to MEDOID_MAX is compared)."""
    flat = np.stack([x.reshape(-1) for x in xs])
    pool = np.random.default_rng(0).choice(len(flat), min(len(flat), MEDOID_MAX), replace=False)
    d = np.linalg.norm(flat[pool][:, None] - flat[pool][None], axis=-1).sum(1)
    return int(pool[d.argmin()])


def arm_frame() -> dict:
    """A real clip frame with the signer's right hand raised in front of the right shoulder and the left hand down."""
    best, best_score = None, np.inf
    for p in sorted(SIGNS.glob("*.json"))[:600]:
        if p.name.startswith("_"):
            continue
        clip = json.loads(p.read_text(encoding="utf-8"))
        for f in clip["frames"]:
            pose = np.asarray(f["pose"])
            sh_r, el_r, wr_r, wr_l = pose[12], pose[14], pose[16], pose[15]
            # world y grows downward: wrist ~15 cm below shoulder height, at the shoulder's x, in front of it
            # (z toward the camera is negative); left wrist hanging low; elbow below the wrist
            score = (abs(wr_r[1] - (sh_r[1] + 0.15)) + abs(wr_r[0] - sh_r[0]) + max(0.0, wr_r[2] - sh_r[2] + 0.15)
                     + max(0.0, 0.25 - (wr_l[1] - sh_r[1])) + max(0.0, wr_r[1] - el_r[1] + 0.05))
            if score < best_score:
                best, best_score = f, score
    return {"pose": best["pose"], "pose2d": best["pose2d"], "left": None, "right": None, "leftW": None}


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=0, help="limit parquet files (0 = all present)")
    a = ap.parse_args()
    ck = torch.load(DEFAULT_MODEL, map_location="cpu", weights_only=True)
    chars = ck["chars"]
    model = SpellNet(len(chars))
    model.load_state_dict(ck["state"])
    model.eval()
    meta = pd.read_csv(FS / "train.csv")
    samples: dict[str, list] = defaultdict(list)
    motion: dict[str, list] = defaultdict(list)
    for p in sorted(FS.glob("*.parquet"))[: a.files or None]:
        used, total = collect(model, chars, meta, p, samples, motion)
        print(f"{p.name}: {used}/{total} sequences read exactly right", flush=True)
    base = arm_frame()
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for c in [chr(i) for i in range(ord("A"), ord("Z") + 1)]:
        if c in MOVING and len(motion[c]) >= 3:
            hands = motion[c][medoid([m[-1] for m in motion[c]])]
        elif samples[c]:
            # the most typical of the samples the reader was surest of (top half): unsure frames are mid-transition or
            # badly tracked, e.g. MediaPipe opens tight fists, which made A and S look loose
            sure = sorted(samples[c], key=lambda x: -x[1])[: max(3, len(samples[c]) // 2)]
            hands = [sure[medoid([h for h, _ in sure])][0]] * HOLD
        else:
            print(f"{c}: no samples")
            continue
        frames = [{**base, "rightW": h.tolist()} for h in hands]
        (OUT / f"{c}.json").write_text(json.dumps({"fps": FPS, "frames": frames}), encoding="utf-8")
        report[c] = len(samples[c])
    print("letters written (samples each):", report)


if __name__ == "__main__":
    main()
