"""Synthetic landmark frames for tests."""

import numpy as np

from server.protocol import Frame


def make_frame(t=0.0, left=True, right=True, shoulder_w=0.2, scale=1.0, jitter=0.0):
    """Upright signer: shoulders at y=0.6, hands near chest. `scale` scales around the shoulder midpoint."""
    mid = np.array([0.5, 0.6])

    def pt(x, y):
        p = mid + (np.array([x, y]) - mid) * scale
        return [float(p[0]), float(p[1]), 0.0]

    pose = [pt(0.5, 0.6) + [1.0] for _ in range(33)]
    pose[0] = pt(0.5, 0.4) + [1.0]
    pose[11] = pt(0.5 + shoulder_w / 2, 0.6) + [1.0]
    pose[12] = pt(0.5 - shoulder_w / 2, 0.6) + [1.0]
    pose[13], pose[14] = pt(0.65, 0.75) + [1.0], pt(0.35, 0.75) + [1.0]
    pose[15], pose[16] = pt(0.6, 0.55) + [1.0], pt(0.4, 0.55) + [1.0]

    def hand(cx):
        return [pt(cx + 0.01 * (i % 5) + jitter, 0.55 - 0.01 * (i // 5)) for i in range(21)]

    return Frame(t=t, left=hand(0.6) if left else None, right=hand(0.4) if right else None, pose=pose)
