"""Audit how well MediaPipe's world landmarks agree with the image landmarks about where the wrists are near the face.

The avatar places a wrist from the world landmarks. If they disagree with the 2D image (which reads across/up well)
by more than 0.3 shoulder widths near the face, Task 7 needs a 2D-based path. Writes reports/motion/placement_audit.json.

Run: uv run python -m training.audit_placement
"""

import json
import math
from pathlib import Path

import numpy as np

from training.build_motion_library import PUBLIC

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "motion" / "placement_audit.json"
FACE_FAR = 1.3  # shoulder widths, as in rocketbox.ts
LIMIT = 0.3
WRISTS = (15, 16)


def _mid(p, a, b):
    return (np.asarray(p[a], float) + np.asarray(p[b], float)) / 2


def aspect_from_pose(pose: list, pose2d: list) -> float:
    """Image W/H from one frame: (|dy2d|/|dyw|) / (|dx2d|/|dxw|), dx the shoulder width, dy shoulders to hips."""
    dxw = abs(pose[11][0] - pose[12][0])
    dx2 = abs(pose2d[11][0] - pose2d[12][0])
    dyw = abs(_mid(pose, 11, 12)[1] - _mid(pose, 23, 24)[1])
    dy2 = abs(_mid(pose2d, 11, 12)[1] - _mid(pose2d, 23, 24)[1])
    return (dy2 / dyw) / (dx2 / dxw)


def _wrist_vs_nose(frame, aspect: float, i: int) -> tuple[np.ndarray, np.ndarray]:
    """(across, up) of wrist i relative to the nose in shoulder widths: from world landmarks, and from aspect-corrected 2D."""
    p, q = frame["pose"], frame["pose2d"]
    sw = abs(p[11][0] - p[12][0])
    sw2 = abs(q[11][0] - q[12][0]) * aspect  # image height units
    w = np.array([p[i][0] - p[0][0], -(p[i][1] - p[0][1])]) / sw
    d = np.array([(q[i][0] - q[0][0]) * aspect, -(q[i][1] - q[0][1])]) / sw2
    return w, d


def placement_error(frame, aspect: float) -> dict[str, float]:
    """Per wrist ("15", "16"): distance in shoulder widths between the world and the 2D across/up offset from the nose."""
    return {str(i): float(np.linalg.norm(np.subtract(*_wrist_vs_nose(frame, aspect, i)))) for i in WRISTS}


def audit_clip(frames: list) -> tuple[float, int]:
    """(worst error over frames with a wrist within FACE_FAR of the nose, number of such frames)."""
    aspect = float(np.median([aspect_from_pose(f["pose"], f["pose2d"]) for f in frames]))
    worst, n = 0.0, 0
    for f in frames:
        err = placement_error(f, aspect)
        for i in WRISTS:
            w, _ = _wrist_vs_nose(f, aspect, i)
            if np.linalg.norm(w) <= FACE_FAR:
                worst, n = max(worst, err[str(i)]), n + 1
    return worst, n


def main() -> None:
    rows = {}
    for path in sorted((PUBLIC / "signs").glob("*.json")):
        try:
            worst, n = audit_clip(json.loads(path.read_text())["frames"])
        except (KeyError, ZeroDivisionError, ValueError):
            continue
        if n and math.isfinite(worst):
            rows[path.stem] = worst
    worst30 = sorted(rows.items(), key=lambda kv: -kv[1])[:30]
    over = sum(v > LIMIT for v in rows.values())
    report = {"clips_with_face_frames": len(rows), "over_limit": over, "limit_shoulder_widths": LIMIT,
              "share_over_limit": over / max(1, len(rows)), "median_worst_error": float(np.median(list(rows.values()))),
              "worst_30": [{"gloss": g, "error": round(v, 3)} for g, v in worst30]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "worst_30"}, indent=1))


if __name__ == "__main__":
    main()
