"""Emotion and intensity from the lower face (mouth, cheeks, eyes) plus how fast and big the signing is.

Never reads brow blendshapes: brows carry grammar (grammar.py), and reading them here would turn questions into "surprise".
"""

import numpy as np

from .protocol import Frame
from .session import Baseline, Emotion

EMO_T = 0.3  # least share of a full expression (FULL) that is a feeling, not a resting face
MIN_FACE = 0.3
# Each shape is read above the signer's resting face (Baseline.face). mouthPucker (resting 0.67 in the median logged
# frame) and eyeSquint (0.23) are left out: they made a resting face half of sad and angry. Lips pressed alone stay
# under EMO_T: the ASL mouth morpheme MM presses them with no feeling at all.
FACE_KEYS = {
    "happy": {"mouthSmileLeft": 1.0, "mouthSmileRight": 1.0, "cheekSquintLeft": 0.5, "cheekSquintRight": 0.5},
    "sad": {"mouthFrownLeft": 1.0, "mouthFrownRight": 1.0},
    "angry": {"noseSneerLeft": 1.0, "noseSneerRight": 1.0, "mouthPressLeft": 0.15, "mouthPressRight": 0.15},
}
# The weighted mean that counts as a full expression. MediaPipe moves these shapes very unevenly: a smile reaches 0.9,
# while frown and sneer stayed under 0.09 in all 978 face frames of 47 logged turns, so one threshold for all three
# meant only happy ever passed. Frown and sneer fulls are starting values, not yet tuned on recorded sad/angry faces.
FULL = {"happy": 0.5, "sad": 0.25, "angry": 0.25}


def face_scores(blend: dict[str, float], rest: dict[str, float]) -> dict[str, float]:
    """Each feeling's share of a full expression in one frame (1.0 = full; may pass 1), above the resting face."""
    return {label: sum(max(0.0, blend.get(k, 0.0) - rest.get(k, 0.0)) * wt for k, wt in keys.items())
            / sum(keys.values()) / FULL[label] for label, keys in FACE_KEYS.items()}


def signing_stats(frames: list[Frame]) -> tuple[float, float]:
    """(mean speed, bounding-box area) of the more active wrist, in shoulder widths."""
    if len(frames) < 2:
        return 0.0, 0.0
    w = max(float(np.median([np.linalg.norm(np.subtract(f.pose[11][:2], f.pose[12][:2])) for f in frames])), 1e-3)
    best = (0.0, 0.0)
    for idx in (15, 16):
        p = np.array([f.pose[idx][:2] for f in frames]) / w
        speed = float(np.linalg.norm(np.diff(p, axis=0), axis=1).mean())
        size = float(np.prod(p.max(0) - p.min(0)))
        best = max(best, (speed, size))
    return best


def read_emotion(frames: list[Frame], baseline: Baseline) -> Emotion:
    faces = [f.blend for f in frames if f.blend is not None]
    if not frames or len(faces) < MIN_FACE * len(frames):
        return {"label": "neutral", "intensity": 0.0}
    each = [face_scores(f, baseline.face) for f in faces]
    scores = {label: float(np.mean([s[label] for s in each])) for label in FACE_KEYS}
    label = max(scores, key=scores.get)
    if scores[label] < EMO_T:
        return {"label": "neutral", "intensity": 0.0}
    speed, size = signing_stats(frames)
    rel = np.mean([speed / baseline.speed if baseline.speed else 1.0,
                   np.sqrt(size / baseline.size) if baseline.size else 1.0])
    # Bigger, faster signing intensifies happy/angry; smaller, slower intensifies sad.
    motion = (1.0 - rel) if label == "sad" else (rel - 1.0)
    intensity = scores[label] * (1.0 + 0.5 * float(np.clip(motion, -1.0, 1.0)))
    return {"label": label, "intensity": float(np.clip(intensity, 0.0, 1.0))}
