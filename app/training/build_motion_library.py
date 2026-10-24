"""Reference videos -> avatar motion clips in web/public/signs/{GLOSS}.json. (Letter clips: training/train_fingerspell.py.)

Clip: {"fps", "frames": [{"pose": 33x[x,y,z] world, "pose2d": 33x[x,y], "left": 21x[x,y,z]|null, "right": ...}]},
trimmed to the sign (wrists above the rest line) and lightly smoothed. Also writes signs/_REST.json (hands down).
Run: uv run python -m training.build_motion_library
"""

import csv
import json
from pathlib import Path

import numpy as np

from server.protocol import Frame
from server.vocab import load_vocab
from training.hand_cleanup import clean_track
from training.landmarks import video_to_frames

ROOT = Path(__file__).resolve().parent.parent
PUBLIC = ROOT / "web" / "public"
ASL_CITIZEN = ROOT / "data" / "asl_citizen" / "ASL_Citizen"
REST_MARGIN = 0.05


def trim_to_sign(frames: list[Frame]) -> tuple[int, int]:
    """[start, end) of frames where a wrist is above the rest line set by the first few frames."""
    wy = np.array([[f.pose[15][1], f.pose[16][1]] for f in frames])
    rest_y = wy[:5].max(1).mean() - REST_MARGIN
    raised = np.where((wy < rest_y).any(1))[0]
    return (int(raised[0]), int(raised[-1]) + 1) if len(raised) else (0, len(frames))


def smooth(seq: list, k: int = 5) -> list:
    """Centred moving average over time for a list of equally-shaped nested lists."""
    a = np.asarray(seq, dtype=np.float64)
    pad = k // 2
    padded = np.concatenate([a[:1].repeat(pad, 0), a, a[-1:].repeat(pad, 0)])
    kernel = np.ones(k) / k
    out = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="valid"), 0, padded)
    return out.tolist()


def _filled(track: list) -> list:
    """Gaps filled with the last seen hand (the first seen before it appears), smoothed; all None if never seen."""
    last, out = None, []
    for h in track:
        last = h if h is not None else last
        out.append(last)
    first = next((h for h in out if h is not None), None)
    return smooth([h if h is not None else first for h in out]) if first is not None else [None] * len(track)


def by_wrist(f: Frame, hw: dict | None = None) -> tuple[dict, dict]:
    """The frame's hands keyed by the signer's own side, decided by which pose wrist each hand sits on (pose 15 = the
    signer's left). MediaPipe's handedness labels disagree with that on ~22% of frames of these videos, and a track
    that flips mid-clip would be smoothed into a blend of both hands."""
    p = np.asarray(f.pose, dtype=np.float64)[:, :2]
    img: dict = {"left": None, "right": None}
    wld: dict = {"left": None, "right": None}
    for label in ("left", "right"):
        h = getattr(f, label)
        if h is None:
            continue
        w = np.asarray(h[0][:2])
        side = "left" if np.linalg.norm(w - p[15]) < np.linalg.norm(w - p[16]) else "right"
        if img[side] is not None:  # both hands nearest one wrist (crossed hands): keep the label's side
            side = "right" if side == "left" else "left"
        img[side], wld[side] = h, (hw or {}).get(label)
    return img, wld


def clip_json(frames: list[Frame], world: list, fps: float, hands_world: list | None = None, clean: bool = False) -> str:
    """Clip hands are keyed by the signer's own side ("left" = the signer's left hand). hands_world (per frame
    {"left", "right"} hand world landmarks, metres, under the detector's labels) adds leftW/rightW, which the avatar's
    finger solver prefers over the image landmarks; clean runs training.hand_cleanup on them (default: gaps held)."""
    s, e = trim_to_sign(frames)
    frames, world = frames[s:e], world[s:e]
    hw = hands_world[s:e] if hands_world is not None else [None] * len(frames)
    sides = [by_wrist(f, h) for f, h in zip(frames, hw)]
    pose = smooth([[p[:3] for p in w] for w in world])
    pose2d = smooth([[p[:2] for p in f.pose] for f in frames])
    hands = {side: _filled([img[side] for img, _ in sides]) for side in ("left", "right")}
    if hands_world is not None:
        fill = (lambda tr: clean_track(tr, fps)) if clean else _filled
        hands |= {side + "W": fill([wld[side] for _, wld in sides]) for side in ("left", "right")}
    return json.dumps({"fps": fps, "frames": [
        {"pose": pose[i], "pose2d": pose2d[i], **{k: v[i] for k, v in hands.items()}} for i in range(len(frames))]})


def _reference_videos(vocab) -> dict[str, Path]:
    """First train video per active gloss."""
    want = {s.asl_citizen.upper(): s.gloss for s in vocab.active}
    out: dict[str, Path] = {}
    with open(ASL_CITIZEN / "splits" / "train.csv", newline="") as fh:
        for row in list(csv.reader(fh))[1:]:
            g = want.get(row[2].strip().upper())
            if g and g not in out:
                out[g] = ASL_CITIZEN / "videos" / row[1]
    return out


def build(name: str, video: Path, out_dir: Path, rest: bool = False) -> bool:
    world: list = []
    frames = video_to_frames(video, world_out=world)
    if len(frames) < 4:
        print(f"skip {name}: no person found")
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(clip_json(frames, world, fps=30.0))
    if rest:
        (out_dir / "_REST.json").write_text(json.dumps({"fps": 30.0, "frames": [
            {"pose": [p[:3] for p in world[0]], "pose2d": [p[:2] for p in frames[0].pose], "left": frames[0].left, "right": frames[0].right}]}))
    return True


def main() -> None:
    vids = _reference_videos(load_vocab())
    ok = sum(build(g, v, PUBLIC / "signs", rest=(i == 0)) for i, (g, v) in enumerate(vids.items()))
    print(f"{ok}/{len(vids)} clips written to {PUBLIC / 'signs'}")


if __name__ == "__main__":
    main()
