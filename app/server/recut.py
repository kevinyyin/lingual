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


def _speed(frames: list[Frame]) -> np.ndarray:
    w = np.stack([_wrists(f) for f in frames])
    v = np.r_[0.0, np.linalg.norm(np.diff(w, axis=0), axis=2).max(1)] / np.median([_width(f) for f in frames])
    return np.convolve(v, np.ones(3) / 3, mode="same")


def recut(run: list[Frame], around: list[Frame], model, fill_gaps: bool = False) -> list[tuple[Segment, list[tuple[str, float]]]] | None:
    """run: the frames of consecutive signed pieces; around: the turn's frames, for each piece's context. -> pieces and
    their top 3, or None when there is no cut to make (shorter than a sign, or no candidate cut within MAX_MS).
    fill_gaps: where speed dips are further apart than MAX_MS, any frame between may be a cut. At a webcam's ~7 frames a
    second dips are few (MY NAME: dips at 0.9 and 3.0 s of 3.3, so no cut at all, and the turn read as one sign)."""
    if len(run) < 3 or run[-1].t - run[0].t < MIN_MS:
        return None
    v = _speed(run)
    cand = [0] + [i for i in range(1, len(run) - 1) if v[i] <= v[i - 1] and v[i] <= v[i + 1]] + [len(run)]
    if fill_gaps:  # every other frame: every one made that turn's read take 3 s (each cut pair is a model call)
        cand = sorted({*cand, *(k for a, b in zip(cand, cand[1:]) if run[b - 1].t - run[a].t > MAX_MS for k in range(a + 2, b, 2))})
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
    return [(Segment(frames=run[i:j], kind="sign", context=read[(i, j)][0]), read[(i, j)][1]) for i, j in best[len(run)][1]]
