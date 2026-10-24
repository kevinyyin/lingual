"""Contact weights for the avatar sign clips, from ASL-LEX phonology.

MediaPipe reads where a hand is across and up the frame well, but guesses its depth: a hand that rests on the chest
(MY) or taps the chin (MOTHER) is placed a hand's width in front of the avatar. ASL-LEX 2.0 codes, per sign, whether
the dominant hand touches something and where (Contact, MajorLocation, MinorLocation). For each clip this finds the
frames where the dominant hand stays at that place *across and up* (depth ignored) and writes a 0-1 weight per frame:
`touchL`/`touchR` (on the body or head) or `meetL`/`meetR` (on the other hand), and for body/head contacts the place
itself, `touchAtL`/`touchAtR` (pose world coordinates). The avatar (web/src/avatar/rocketbox.ts) then moves the hand
straight back by that weight until it reaches the surface, and steers it onto its own version of the place. Only these fields are added; re-running
training.clips_from_wlasl drops them, so run this again after it.

Run: uv run python -m training.touch_from_asllex [--only MY,MOTHER]
"""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from training.audit_placement import aspect_from_pose
from training.build_motion_library import PUBLIC

ROOT = Path(__file__).resolve().parent.parent
ASLLEX = ROOT / "data" / "asllex" / "sign_props.json"
CITIZEN_TRAIN = ROOT / "data" / "asl_citizen" / "splits" / "train.csv"
KEYS = ("touchL", "touchR", "meetL", "meetR", "touchAtL", "touchAtR")
# within this many shoulder widths (across and up) of the place counts as there, fading out over FADE more
NEAR = {"Body": 0.35, "Head": 0.25, "Hand": 0.15, "Neutral": 0.15}  # Neutral + contact: the two hands touch
FADE = 0.15
DWELL_S = 0.12  # there for this long either side of a frame: a hand passing on its way elsewhere is not a contact
# ASL-LEX says the contact happens, so a take whose hand holds only *near* the estimated place (MediaPipe's world
# landmarks are loose, the place is an estimate from face/shoulder points) still has it, at its closest hold
REACH = 0.6
SETTLE = 0.08  # beyond that closest hold still counted as there
TIPS = [0, 4, 8, 12, 16, 20]  # wrist and fingertips: the parts of a hand that make contact
# MediaPipe pose: nose, eyes, mouth corners, shoulders, hips
NOSE, EYES, MOUTH, SHOULDERS, HIPS = 0, (2, 5), (9, 10), (11, 12), (23, 24)


def body_xy(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Origin between the shoulders, unit axes across (x) and up the spine (y), and the shoulder width."""
    o = pose[list(SHOULDERS)].mean(0)
    x = pose[SHOULDERS[0]] - pose[SHOULDERS[1]]
    w = max(float(np.linalg.norm(x)), 1e-6)
    x = x / w
    y = o - pose[list(HIPS)].mean(0)
    y = y - x * (y @ x)
    return o, x, y / max(float(np.linalg.norm(y)), 1e-6), w


def anchor(pose: np.ndarray, minor: str, side: int) -> np.ndarray | None:
    """The place ASL-LEX names, in pose world coordinates (depth is never used). `side`: the dominant shoulder index."""
    o, _, up, w = body_xy(pose)
    n, e, m = pose[NOSE], pose[list(EYES)].mean(0), pose[list(MOUTH)].mean(0)
    places = {
        "Forehead": e + (e - m) * 1.2, "HeadTop": e + (e - m) * 2.5, "Eye": e, "CheekNose": n,
        "Mouth": m, "UpperLip": (m + n) / 2, "Chin": m + (m - n) * 2.0, "UnderChin": m + (m - n) * 3.0,
        "Neck": o + up * 0.15 * w, "Clavicle": o - up * 0.12 * w, "TorsoTop": o - up * 0.35 * w,
        "TorsoMid": o - up * 0.7 * w, "TorsoBottom": o - up * 1.0 * w, "Waist": o - up * 1.1 * w, "Hips": o - up * 1.2 * w,
        "Shoulder": pose[side],
    }
    return places.get(minor)


def hand_points(f: dict, side: str) -> np.ndarray | None:
    """Wrist and fingertips in pose world coordinates: the pose wrist plus the hand's own (world) shape."""
    hw = f.get(f"{side}W")
    if not hw:
        return None
    hw = np.asarray(hw, dtype=float)
    wrist = np.asarray(f["pose"][16 if side == "right" else 15], dtype=float)
    return wrist + hw[TIPS] - hw[0]


def weights(d: np.ndarray, near: float, fps: float) -> np.ndarray:
    """Per-frame contact weight from the across/up distance (shoulder widths) between the hand and its place: near
    enough, and staying there (a hold, tap or rub) rather than passing through. The closest approach alone failed:
    MY's hand rests on the chest for a second, but its closest point to the chest centre came as it dropped away."""
    k = max(1, round(DWELL_S * fps))
    pad = lambda x, v: np.pad(x, k, constant_values=v)  # noqa: E731
    win = lambda x: np.lib.stride_tricks.sliding_window_view(x, 2 * k + 1)  # noqa: E731
    d = np.where(np.isfinite(d), d, np.inf)
    hold = win(pad(d, np.inf)).max(1)  # the farthest the hand gets over each dwell window
    if not np.isfinite(hold).any() or hold.min() > REACH:
        return np.zeros_like(d)
    near = max(near, float(hold.min()) + SETTLE)
    prox = np.clip(1 - (d - near) / FADE, 0, 1)
    w = win(pad(win(pad(prox, 0)).min(1), 0)).max(1)  # opening: drops brief passes, keeps holds their full length
    return np.round(w * w * (3 - 2 * w), 3)  # smoothstep: no step into or out of contact


def dominant(frames: list[dict]) -> str:
    """The right hand unless it is barely tracked: clips keep each hand on the signer's own side, and best takes are
    right-handed first (a two-handed sign's other hand can move as much, so travel doesn't tell them apart)."""
    seen = {s: sum(1 for f in frames if f.get(s)) for s in ("right", "left")}
    return "right" if seen["right"] >= 0.5 * seen["left"] else "left"


# Image-detected face contact, for clips ASL-LEX gives no contact (or no entry): the dominant hand's nearest 2D landmark
# (all 21) within IMG_NEAR shoulder widths (aspect-corrected 2D) of the chin or mouth for the ASL-LEX path's dwell (2 x DWELL_S + one frame) in a row.
# MediaPipe's world pose put such a hand a palm in front of the face (GRANDMOTHER); the image shows it at the chin.
IMG_NEAR, IMG_RAMP = 0.10, 3  # weights ramp over the first/last IMG_RAMP-1 frames of a hold


def face_points_2d(q: np.ndarray) -> list[np.ndarray]:
    """Mouth (midpoint of the corners) and chin: the mouth moved on, away from the nose, by 0.35x the nose-to-mouth distance."""
    m = q[list(MOUTH)].mean(0)
    return [m, m + (m - q[NOSE]) * 0.35]


def image_contact(clip: dict) -> tuple[str, str]:
    frames = clip["frames"]
    side = dominant(frames)
    s = "R" if side == "right" else "L"
    try:
        aspect = float(np.median([aspect_from_pose(f["pose"], f["pose2d"]) for f in frames]))
    except (KeyError, ZeroDivisionError, ValueError):
        return "no contact", ""
    if not np.isfinite(aspect):
        return "no contact", ""
    n = len(frames)
    need = 2 * max(1, round(DWELL_S * float(clip.get("fps") or 30))) + 1  # the ASL-LEX path's dwell window
    d = np.full((n, 2), np.inf)  # per frame: distance (shoulder widths) to the mouth, to the chin
    pts = np.full((n, 2, 3), np.nan)  # and those face points in pose world coordinates
    for i, f in enumerate(frames):
        hand = f.get(side)
        if not hand or not f.get("pose2d"):
            continue
        scale = np.array([aspect, 1.0])
        q = np.asarray(f["pose2d"], dtype=float)[:, :2] * scale
        sw2 = abs(q[SHOULDERS[0], 0] - q[SHOULDERS[1], 0])
        if sw2 < 1e-6:
            continue
        h = np.asarray(hand, dtype=float)[:, :2] * scale
        p = np.asarray(f["pose"], dtype=float)
        k = abs(p[SHOULDERS[0], 0] - p[SHOULDERS[1], 0]) / sw2
        for j, c in enumerate(face_points_2d(q)):
            d[i, j] = np.linalg.norm(h - c, axis=-1).min() / sw2
            # the nose plus the 2D offset scaled to world metres; depth is the nose's
            pts[i, j] = [p[NOSE, 0] + (c - q[NOSE])[0] * k, p[NOSE, 1] + (c - q[NOSE])[1] * k, p[NOSE, 2]]
    near = d.min(1) < IMG_NEAR
    w = np.zeros(n)
    at: list[np.ndarray | None] = [None] * n
    i = 0
    while i < n:
        if not near[i]:
            i += 1
            continue
        j = i
        while j < n and near[j]:
            j += 1
        if j - i >= need:
            m = int(np.argmin(np.median(d[i:j], axis=0)))  # mouth or chin, once per hold, and one place for all of it
            place = np.median(pts[i:j, m], axis=0)
            w[i:j] = [min(t + 1, j - i - t, IMG_RAMP) / IMG_RAMP for t in range(j - i)]
            for t in range(i, j):
                at[t] = place
        i = j
    if not w.any():
        return "no contact", ""
    for f, v, a in zip(frames, w, at):
        if v > 0:
            f[f"touch{s}"] = round(float(v), 3)
            f[f"touchAt{s}"] = [round(float(x), 4) for x in a]
    return "image", f"touch{s} on {int((w > 0).sum())}/{n} frames"


BASE_TYPES = ("AsymmetricalDifferentHandshape", "AsymmetricalSameHandshape")


def mark_base(clip: dict, props: dict) -> None:
    """`base`: the hand ASL-LEX makes the stationary base of a two-handed sign (its sign type is asymmetrical: the other
    hand acts on it, which keeps one handshape). The avatar holds that hand in one clean shape (web/src/avatar/baseHand.ts):
    in the videos the acting hand covers it and the tracker guesses its fingers (NICE)."""
    clip.pop("base", None)
    frames = clip["frames"]
    two = all(sum(1 for f in frames if f.get(s)) >= len(frames) / 2 for s in ("left", "right"))
    if props and props.get("SignType.2.0") in BASE_TYPES and two:
        clip["base"] = "L" if dominant(frames) == "right" else "R"


def annotate(clip: dict, props: dict) -> tuple[str, str]:
    """Add contact weights to `clip` in place: from ASL-LEX where it codes a contact the hand makes, else from the
    image (face contact). Returns (outcome, detail) for the log."""
    outcome = annotate_lex(clip, props)
    if outcome[0] == "touch" or outcome[0].startswith("meet"):
        return outcome
    for f in clip["frames"]:  # an ASL-LEX attempt that found nothing leaves nothing behind
        for k in KEYS:
            f.pop(k, None)
    if props and props.get("MajorLocation.2.0") != "Head":  # a sign ASL-LEX places elsewhere stays where it is
        return outcome
    img = image_contact(clip)
    return img if img[0] == "image" else outcome


def annotate_lex(clip: dict, props: dict) -> tuple[str, str]:
    frames = clip["frames"]
    for f in frames:
        for k in KEYS:
            f.pop(k, None)
    if not props or props.get("Contact.2.0") != 1 or len(frames) < 2:
        return "no contact", ""
    major, minor = props.get("MajorLocation.2.0"), props.get("MinorLocation.2.0")
    side = dominant(frames)
    other = "left" if side == "right" else "right"
    s = "R" if side == "right" else "L"
    d = np.full(len(frames), np.inf)
    at: list[list[float] | None] = [None] * len(frames)
    for i, f in enumerate(frames):
        pose = np.asarray(f["pose"], dtype=float)
        pts = hand_points(f, side)
        if pts is None:
            continue
        o, x, y, w = body_xy(pose)
        flat = lambda p: np.stack([(p - o) @ x, (p - o) @ y], -1) / w  # noqa: E731 - across/up, shoulder widths
        if major in ("Body", "Head"):
            a = anchor(pose, minor, 12 if side == "right" else 11)
            if a is None:
                return "place not mapped", f"{major}/{minor}"
            at[i] = [round(float(v), 4) for v in a]
            d[i] = np.linalg.norm(flat(pts) - flat(a), axis=-1).min()
        elif major in ("Hand", "Neutral"):
            q = hand_points(f, other)
            if q is not None:
                d[i] = np.linalg.norm(flat(pts)[:, None] - flat(q)[None], axis=-1).min()
    if major not in NEAR:
        return "place not mapped", f"{major}/{minor}"
    key = f"meet{s}" if major in ("Hand", "Neutral") else f"touch{s}"
    w = weights(d, NEAR[major], float(clip.get("fps") or 30))
    if not w.any():
        return "hand never reaches it", f"{major}/{minor}"
    for f, v, a in zip(frames, w, at):
        if v > 0:
            f[key] = float(v)
            if a is not None:
                f[f"touchAt{s}"] = a
    return key[:5], f"{major}/{minor} {key} on {int((w > 0.5).sum())}/{len(frames)} frames"


def lexicon(vocab_glosses: list[str]) -> dict[str, dict]:
    """Gloss -> ASL-LEX entry: by the ASL Citizen gloss's ASL-LEX code, else by the English entry name."""
    entries = json.loads(ASLLEX.read_text(encoding="utf-8"))
    by_code = {e["Code"]: e for e in entries}
    by_name: dict[str, dict] = {}
    for e in entries:
        by_name.setdefault(re.sub(r"_\d+$", "", e["EntryID"]).upper().replace("_", "-"), e)
    code_of: dict[str, str] = {}
    if CITIZEN_TRAIN.exists():
        with open(CITIZEN_TRAIN, newline="", encoding="utf-8") as fh:
            for row in list(csv.reader(fh))[1:]:
                code_of.setdefault(re.sub(r"\d+$", "", row[2].strip()).upper(), row[3].strip())
    out = {}
    for g in vocab_glosses:
        e = by_code.get(code_of.get(g, "")) or by_name.get(g)
        if e:
            out[g] = e
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated glosses")
    ap.add_argument("--dir", default=str(PUBLIC / "signs"))
    args = ap.parse_args()
    paths = sorted(p for p in Path(args.dir).glob("*.json") if not p.stem.startswith("_"))
    if args.only:
        want = {g.strip().upper() for g in args.only.split(",")}
        paths = [p for p in paths if p.stem in want]
    lex = lexicon([p.stem for p in paths])
    counts: dict[str, int] = {}
    gained: list[str] = []
    bases = 0
    for p in paths:
        clip = json.loads(p.read_text(encoding="utf-8"))
        outcome, detail = annotate(clip, lex.get(p.stem, {}))
        mark_base(clip, lex.get(p.stem, {}))
        bases += "base" in clip
        counts[outcome] = counts.get(outcome, 0) + 1
        if outcome == "image":
            gained.append(p.stem)
        if args.only:
            print(f"{p.stem}: {outcome} {detail}")
        p.write_text(json.dumps(clip, separators=(",", ":")), encoding="utf-8")
    print(f"{len(gained)} clips gained image contact; first 30: {', '.join(gained[:30])}")
    print(f"{bases} clips marked with a stationary base hand (ASL-LEX asymmetrical sign type)")
    print(f"{len(paths)} clips, {len(lex)} found in ASL-LEX: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
