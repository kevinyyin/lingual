"""Look-alike signs the sign model mixes up, told apart by the one thing that differs between them.

MY and PLEASE: a flat hand on the chest; PLEASE rubs a circle there, MY does not. A webcam signer's MY read as PLEASE
(0.57-0.96, MY second) in every turn. The palm's spread over the sign's middle (shoulder widths) separates them cleanly:
WLASL PLEASE 0.11-0.27, WLASL MY 0.008-0.043, the webcam signer's MY 0.009-0.043.
"""

import numpy as np

from .protocol import Frame

PLEASE_SPREAD = 0.07  # palm spread (shoulder widths) above which the hand circles: PLEASE; below: MY
# MY is a flat hand held still on the chest. The sign model read a webcam signer's MY as PLEASE, LIKE or HUNGRY with MY
# often not in its first three, so the shape decides: still (spread), flat (fingers' mean bend, degrees), on the chest
# (palm centre from the shoulders' midpoint, shoulder widths). At the webcam's ~7 frames a second it held for the
# signer's MY 8 of 8, WLASL MY/MINE 4 of 5, and none of 45 WLASL chest and nearby signs (PLEASE, LIKE, HUNGRY, SORRY,
# FEEL, ME, I, YOUR, FINE, EXCITED, THANK-YOU, GOOD, WANT, NAME).
MY_SPREAD, MY_CURL, MY_X, MY_Y = 0.05, 30.0, 0.15, (0.1, 0.5)


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


def _curl(h) -> float:
    """Mean bend of the four fingers (degrees, image landmarks): 0 straight; a fist ~200."""
    p = np.asarray(h, dtype=float)[:, :2]
    bends = []
    for c in ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)):
        bones = [p[c[0]] - p[0]] + [p[c[i + 1]] - p[c[i]] for i in range(3)]
        cos = [np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9) for a, b in zip(bones, bones[1:])]
        bends.append(sum(np.degrees(np.arccos(np.clip(cos, -1, 1)))))
    return float(np.mean(bends))


def held_on_chest(frames: list[Frame]) -> bool:
    """A flat hand held still on the middle of the chest over the sign's middle 60%: MY (MY_*)."""
    rows = []
    for f in frames:
        h = f.right if f.right is not None else f.left
        if h is None:
            continue
        sw = abs(f.pose[11][0] - f.pose[12][0]) or 1.0
        cx, sy = (f.pose[11][0] + f.pose[12][0]) / 2, (f.pose[11][1] + f.pose[12][1]) / 2
        rows.append(((h[9][0] - cx) / sw, (h[9][1] - sy) / sw, _curl(h)))
    if len(rows) < 4:
        return False
    r = np.asarray(rows)
    n = len(r)
    if n >= 8:
        r = r[n // 5: n - n // 5]
    x, y, curl = (float(np.median(r[:, k])) for k in range(3))
    return (float(np.linalg.norm(r[:, :2].std(0))) < MY_SPREAD and curl < MY_CURL and abs(x) < MY_X
            and MY_Y[0] < y < MY_Y[1])


def settle(frames: list[Frame], top: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """`top` with MY first for a flat hand held on the chest (held_on_chest), else with MY or PLEASE first by the palm's
    circling when both are among the first three guesses."""
    if top and top[0][0] != "MY" and held_on_chest(frames):
        return [("MY", top[0][1])] + [(g, p) for g, p in top if g != "MY"]
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
