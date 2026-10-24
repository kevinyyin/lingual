"""Avatar sign clips from the asl-spike's already-extracted ASL Citizen landmarks (no videos needed).

The spike ran MediaPipe Holistic over ASL Citizen and saved cache/s2/citizen/<video>.npz (pose, pose_world, lh, rh,
fps); results/s2_citizen_subset.csv maps each video to its gloss. This writes the same clips as
training.build_motion_library (same trimming and smoothing via clip_json) into web/public/signs/.

Run: uv run python -m training.clips_from_spike [--npz-dir ../asl-spike/cache/s2/citizen] [--check]
ASL Citizen-derived: non-commercial research use only; keep the clips private.
"""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from server.protocol import Frame
from server.vocab import load_vocab
from training.build_motion_library import PUBLIC, clip_json

SPIKE = Path(__file__).resolve().parent.parent.parent / "asl-spike"
SUBSET = SPIKE / "results" / "s2_citizen_subset.csv"


# ASL Citizen names for the same sign under another English word (index to chest; wave; brush of the palm).
ALIASES = {"ME": "I", "BYE": "GOODBYE", "EXCUSE": "EXCUSE-ME"}


def _match(citizen: str, want: dict[str, str]) -> tuple[str | None, int]:
    """ASL Citizen gloss -> (our gloss, variant rank). ASL Citizen numbers variants (WANT1, WANT2) and spells some
    (W.H.A.T); the plain name or variant 1 is preferred."""
    name = citizen.strip().upper()
    base = re.sub(r"\d+$", "", name.replace(".", ""))
    g = want.get(name) or want.get(base) or want.get(ALIASES.get(base, ""))
    return g, 0 if name in (base, base + "1") else 2 if "." in name else 1


def reference_npz(vocab, npz_dir: Path) -> tuple[dict[str, Path], list[str]]:
    """First train-split (else any split) landmark file per active gloss, and the glosses the spike lacks."""
    want = {s.asl_citizen.upper(): s.gloss for s in vocab.active} | {s.gloss: s.gloss for s in vocab.active}
    rank = {"train": 0, "val": 1, "test": 2}
    best: dict[str, tuple[tuple[int, int], Path]] = {}
    with open(SUBSET, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            g, variant = _match(row["Gloss"], want)
            path = npz_dir / f"{Path(row['Video file']).stem}.npz"
            r = (variant, rank.get(row["split"], 3))
            if g and path.exists() and (g not in best or r < best[g][0]):
                best[g] = (r, path)
    found = {g: p for g, (_, p) in best.items()}
    return found, sorted(s.gloss for s in vocab.active if s.gloss not in found)


def _hand(h: np.ndarray) -> list | None:
    return None if np.isnan(h).any() else h.tolist()


def npz_frames(path: Path) -> tuple[list[Frame], list, float]:
    """Frames with hands keyed by the signer's own side, by the pose wrist each hand sits on (pose 16 = the signer's
    right wrist), rather than trusting either detector's left/right labels (clip_json re-checks with by_wrist)."""
    with np.load(path) as z:
        pose, world, lh, rh, fps = z["pose"], z["pose_world"], z["lh"], z["rh"], float(z["fps"]) or 30.0
    frames, worlds = [], []
    for i in range(len(pose)):
        if np.isnan(pose[i]).any() or np.isnan(world[i]).any():
            continue
        f = Frame(t=i * 1000.0 / fps, pose=np.nan_to_num(pose[i]).tolist())
        for h in (lh[i], rh[i]):
            pts = _hand(h)
            if pts is None:
                continue
            d_right = np.linalg.norm(h[0, :2] - pose[i, 16, :2])  # signer's right wrist
            d_left = np.linalg.norm(h[0, :2] - pose[i, 15, :2])
            if d_right < d_left:
                f.right = pts
            else:
                f.left = pts
        frames.append(f)
        worlds.append(world[i].tolist())
    return frames, worlds, fps


def signs2_frames(entry: dict) -> tuple[list[Frame], list, float]:
    """One sign of the spike's avatar/signs2.json ({"fps", "frames": [{"pw", "p2", "lh", "rh"}]}, null = missing;
    x pre-scaled by the video's aspect ratio, which keeps hand and arm angles true) -> frames + world pose."""
    frames, worlds = [], []
    fps = float(entry.get("fps") or 30.0)
    for i, fr in enumerate(entry["frames"]):
        if fr["p2"] is None or fr["pw"] is None:
            continue
        f = Frame(t=i * 1000.0 / fps, pose=[[*p[:3], 1.0] for p in fr["p2"]])
        for hand in (fr["lh"], fr["rh"]):
            if hand is None:
                continue
            wrist = np.asarray(hand[0][:2])
            if np.linalg.norm(wrist - fr["p2"][16][:2]) < np.linalg.norm(wrist - fr["p2"][15][:2]):
                f.right = hand  # on the signer's right wrist (see npz_frames)
            else:
                f.left = hand
        frames.append(f)
        worlds.append(fr["pw"])
    return frames, worlds, fps


def _write(out: Path, gloss: str, frames: list[Frame], world: list, fps: float) -> bool:
    if len(frames) < 4:
        print(f"skip {gloss}: no person found")
        return False
    (out / f"{gloss}.json").write_text(clip_json(frames, world, fps), encoding="utf-8")
    if not (out / "_REST.json").exists():
        (out / "_REST.json").write_text(json.dumps({"fps": fps, "frames": [{
            "pose": [p[:3] for p in world[0]], "pose2d": [p[:2] for p in frames[0].pose],
            "left": frames[0].left, "right": frames[0].right}]}), encoding="utf-8")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", type=Path, default=SPIKE / "cache" / "s2" / "citizen")
    ap.add_argument("--signs2", type=Path, help="build from the spike's avatar/signs2.json instead of the npz cache")
    ap.add_argument("--check", action="store_true", help="only report which vocabulary signs the spike covers")
    a = ap.parse_args()
    vocab = load_vocab()
    out = PUBLIC / "signs"
    if a.signs2:
        from server.spike_classifier import RENAME

        if not a.signs2.exists():
            print(f"{a.signs2} not found: ask for asl-spike/avatar/signs2.json (built by spike2/v2_lexicon.py; local only)")
            return
        lex = json.loads(a.signs2.read_text(encoding="utf-8"))["signs"]
        refs = {RENAME.get(c, c): e for c, e in lex.items() if RENAME.get(c, c) in vocab.glosses}
        missing = sorted(vocab.glosses - set(refs))
        print(f"{len(refs)}/{len(vocab.active)} vocabulary signs are in {a.signs2}")
        if missing:
            print(f"missing ({len(missing)}): {' '.join(missing)}")
        if a.check:
            return
        out.mkdir(parents=True, exist_ok=True)
        ok = sum(_write(out, g, *signs2_frames(e)) for g, e in refs.items())
        print(f"{ok}/{len(refs)} clips written to {out}")
        return
    refs, missing = reference_npz(vocab, a.npz_dir)
    print(f"{len(refs)}/{len(vocab.active)} vocabulary signs have spike landmarks in {a.npz_dir}")
    if missing:
        print(f"missing ({len(missing)}): {' '.join(missing)}")
    if a.check or not refs:
        return
    out.mkdir(parents=True, exist_ok=True)
    ok = sum(_write(out, g, *npz_frames(p)) for g, p in refs.items())
    print(f"{ok}/{len(refs)} clips written to {out}")


if __name__ == "__main__":
    main()
