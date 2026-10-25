"""Demo mode: a signed turn snapped to the nearest of a few scripted sentences (vocab/demo_sentences.yaml).

300 replayed short sentences (2-4 signs) read exactly right 53-59% of the time, mostly limited by cutting: the turn
had the wrong number of pieces 34-40% of the time (usually two signs merged into one), while each rightly cut piece's
top guess was right 90% of the time and in its top 3 for 97-98%. With the demo's sentences known, the read pieces are
matched to each one, tolerating a merged or split piece; a clear winner is said as its exact English. Otherwise None,
and the turn is translated as usual.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "vocab" / "demo_sentences.yaml"
TOP_K = 10  # guesses per piece matched against: the right sign is in the top 3 for 97% of rightly cut pieces
MISSING_P = 1e-3  # a sentence's sign not among a piece's guesses
MERGE_COST = 1.0  # one piece standing for two signs of the sentence in a row
SPLIT_COST = 1.5  # a piece that stands for no sign of the sentence (one sign cut in two, or a stray piece)
FLOOR = -3.0  # per-sign score (mean log probability, less costs) a snap needs: about one sign of three may be missing
MARGIN = 0.5  # and its lead over the runner-up


@dataclass(frozen=True)
class DemoSentence:
    signs: tuple[str, ...]
    english: str


def load(path: Path | str = DEFAULT_PATH) -> list[DemoSentence]:
    """The demo's sentences; empty (the feature off) without the file."""
    path = Path(path)
    if not path.exists():
        return []
    return [DemoSentence(tuple(str(g).upper() for g in e["signs"]), str(e["english"]))
            for e in yaml.safe_load(path.read_text(encoding="utf-8")) or []]


def _cost(tops: list[list[tuple[str, float]]], signs: tuple[str, ...]) -> float:
    """Alignment cost of the pieces' guesses to a sentence: -log p of each matched sign in its piece; a piece may stand
    for two signs in a row (MERGE_COST) or for none (SPLIT_COST); a sentence sign with no piece costs a missing one."""
    miss = -math.log(MISSING_P)
    nl = [{g: -math.log(max(p, MISSING_P)) for g, p in t[:TOP_K]} for t in tops]
    n, m = len(tops), len(signs)
    inf = float("inf")
    d = [[inf] * (m + 1) for _ in range(n + 1)]
    d[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            c = d[i][j]
            if c == inf:
                continue
            if i < n and j < m:  # piece i is sign j
                d[i + 1][j + 1] = min(d[i + 1][j + 1], c + nl[i].get(signs[j], miss))
            if i < n and j + 1 < m:  # piece i is signs j and j+1 merged
                both = min(nl[i].get(signs[j], miss), nl[i].get(signs[j + 1], miss))
                d[i + 1][j + 2] = min(d[i + 1][j + 2], c + both + MERGE_COST)
            if i < n:  # piece i is extra
                d[i + 1][j] = min(d[i + 1][j], c + SPLIT_COST)
            if j < m:  # sign j has no piece
                d[i][j + 1] = min(d[i][j + 1], c + miss)
    return d[n][m]


def snap(tops: list[list[tuple[str, float]]], sentences: list[DemoSentence]) -> tuple[DemoSentence | None, float, float]:
    """The demo sentence the pieces' guesses match (each piece's ranked (gloss, p)), with its per-sign score and the
    runner-up's; None for the sentence when no match is clear (FLOOR, MARGIN)."""
    if not tops or not sentences:
        return None, -math.inf, -math.inf
    scored = sorted(((-_cost(tops, s.signs) / len(s.signs), s) for s in sentences), key=lambda x: -x[0])
    best, sentence = scored[0]
    runner = scored[1][0] if len(scored) > 1 else -math.inf
    return (sentence if best >= FLOOR and best - runner >= MARGIN else None), best, runner
