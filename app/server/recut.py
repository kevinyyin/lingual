"""The translator's signs re-cut where the sign model is surest. The segmenter cuts at 300 ms pauses, and signs in a
sentence often flow on without one. Every dip in the wrists' speed is a candidate cut; dynamic programming keeps the
cuts whose pieces the model reads most surely, less PIECE_COST each. Replayed sentences (WLASL signs joined, new sign
model): signs wrong 0.70 -> 0.59, exactly right 7% -> 15%, the right number of pieces 44% -> 62%; with 0.3-0.6 s
holds between signs still as good or better (0.77 -> 0.65, 0.74 -> 0.72); ~50 ms a sentence.
"""

import math

import numpy as np

from .protocol import Frame
from .segmenter import _width, _wrists
from .session import Segment

PIECE_COST = 1.0  # log-probability each piece must earn: two sure pieces only beat one when the one is unsure
MIN_MS, MAX_MS = 250.0, 1800.0  # a piece's length
CONTEXT_MS = 250.0  # read with this much either side (sentence replays: 250 ms 50%, 500 ms 48%, none 41%)
# Neighbouring pieces the model reads as the same sign are one sign (a sign's repeated taps, a hold cut in two): teaching
# clips 866 -> 881 of 995 right, none broke. Only inside one of the segmenter's segments, so a sign repeated after a real
# pause (NO, rest, NO) stays two words, and never longer than this.
SAME_MAX_MS = 2500.0


def _speed(frames: list[Frame]) -> np.ndarray:
    w = np.stack([_wrists(f) for f in frames])
    v = np.r_[0.0, np.linalg.norm(np.diff(w, axis=0), axis=2).max(1)] / np.median([_width(f) for f in frames])
    return np.convolve(v, np.ones(3) / 3, mode="same")


def recut(run: list[Frame], around: list[Frame], model, spans: list[tuple[float, float]] = ()
          ) -> list[tuple[Segment, list[tuple[str, float]]]] | None:
    """run: the frames of consecutive signed pieces; around: the turn's frames, for each piece's context; spans: the
    segmenter's segments' (first, last) frame times, where same-sign pieces may merge. -> pieces and their top 3, or None
    when there is no cut to make (shorter than a sign, or no candidate cut within MAX_MS)."""
    if len(run) < 3 or run[-1].t - run[0].t < MIN_MS:
        return None
    v = _speed(run)
    cand = [0] + [i for i in range(1, len(run) - 1) if v[i] <= v[i - 1] and v[i] <= v[i + 1]] + [len(run)]
    read: dict[tuple[int, int], tuple[list[Frame], list[tuple[str, float]]]] = {}
    best: dict[int, tuple[float, list[tuple[int, int]]]] = {0: (0.0, [])}
    for j in cand[1:]:
        for i in cand:
            if i >= j or i not in best or not MIN_MS <= run[j - 1].t - run[i].t <= MAX_MS:
                continue
            if (i, j) not in read:
                ctx = [f for f in around if run[i].t - CONTEXT_MS <= f.t <= run[j - 1].t + CONTEXT_MS]
                read[(i, j)] = (ctx, model.topk(ctx, 3))
            score = best[i][0] + math.log(read[(i, j)][1][0][1] + 1e-6) - PIECE_COST
            if j not in best or score > best[j][0]:
                best[j] = (score, best[i][1] + [(i, j)])
    if len(run) not in best:
        return None
    pieces = [(Segment(frames=run[i:j], kind="sign", context=read[(i, j)][0]), read[(i, j)][1])
              for i, j in best[len(run)][1]]
    return _same_sign_once(pieces, spans)


def _same_sign_once(pieces, spans):
    """Neighbouring pieces read as the same sign merged when the cut between them is inside a segment and together they
    reach no other segment (a piece can run on over a pause: never one word across it)."""
    out = []
    for s, top in pieces:
        if out:
            p, ptop = out[-1]
            cut = s.frames[0].t
            start, end = p.frames[0].t, s.frames[-1].t
            within = [(lo, hi) for lo, hi in spans if lo <= end and hi >= start]  # the segments the two pieces reach
            if (top and ptop and top[0][0] == ptop[0][0] and len(within) == 1 and within[0][0] < cut < within[0][1]
                    and end - start <= SAME_MAX_MS):
                ctx = sorted({f.t: f for f in [*(p.context or []), *(s.context or [])]}.values(), key=lambda f: f.t)
                out[-1] = (Segment(frames=p.frames + s.frames, kind="sign", context=ctx),
                           ptop if ptop[0][1] >= top[0][1] else top)
                continue
        out.append((s, top))
    return out
