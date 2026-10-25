"""Look-alike signs the sign model mixes up, told apart by the one thing that differs between them.

MY and PLEASE: a flat hand on the chest; PLEASE rubs a circle there, MY does not. A webcam signer's MY read as PLEASE
(0.57-0.96, MY second) in every turn. The palm's spread over the sign's middle (shoulder widths) separates them cleanly:
WLASL PLEASE 0.11-0.27, WLASL MY 0.008-0.043, the webcam signer's MY 0.009-0.043.
"""

import numpy as np

from .protocol import Frame

PLEASE_SPREAD = 0.07  # palm spread (shoulder widths) above which the hand circles: PLEASE; below: MY


def palm_spread(frames: list[Frame]) -> float | None:
    """How much the palm centre (middle knuckle) moves about over the middle 60% of the frames with a hand, in
    shoulder widths: not the rise to the chest and the drop from it. None with too few frames."""
    pts = []
    for f in frames:
        h = f.right if f.right is not None else f.left
        if h is None:
            continue
        sw = abs(f.pose[11][0] - f.pose[12][0]) or 1.0
        pts.append([(h[9][0] - (f.pose[11][0] + f.pose[12][0]) / 2) / sw, (h[9][1] - f.pose[11][1]) / sw])
    if len(pts) < 4:
        return None
    p = np.asarray(pts)
    n = len(p)
    if n >= 8:
        p = p[n // 5: n - n // 5]
    return float(np.linalg.norm(p.std(0)))


def settle(frames: list[Frame], top: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """`top` with MY or PLEASE first by the palm's circling, when both are among the first three guesses."""
    head = [g for g, _ in top[:3]]
    if "MY" not in head or "PLEASE" not in head or head[0] not in ("MY", "PLEASE"):
        return top
    spread = palm_spread(frames)
    if spread is None:
        return top
    want = "PLEASE" if spread > PLEASE_SPREAD else "MY"
    if head[0] == want:
        return top
    p = dict(top)
    first = (want, p[head[0]])  # the pair's confidence goes to the one the motion picks
    return [first] + [(g, p[want] if g == head[0] else q) for g, q in top if g != want]


class LookalikeRecognizer:
    """A recognizer whose guesses are settled between look-alike signs (settle); everything else is the wrapped one's."""

    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def topk(self, frames, k=3):
        return settle(frames, self.inner.topk(frames, k))

    def recognize(self, seg):
        return settle(seg.context or seg.frames, self.inner.recognize(seg))
