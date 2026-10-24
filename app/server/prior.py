"""A prior over signs on top of a recognizer: the curated everyday signs (vocab.yaml, ~160) are far likelier in
conversation than the ~2,750 generated ones (vocab_extended.yaml), which the model otherwise picks as often.

training.eval_live, one-sign videos through the live segmenter at the browser's frame rate (ASL Citizen test signers +
WLASL, 603 curated; 200 random generated), sign spoken without a pick, by curated weight:
  1: curated 0.406, generated 0.485 | 30: 0.522, 0.470 | 100: 0.551, 0.455 | 300: 0.564, 0.440
"""

from .protocol import Frame

CURATED_PRIOR = 100.0


class PriorRecognizer:
    def __init__(self, recognizer, vocab, curated: float = CURATED_PRIOR):
        self.r, self.vocab, self.curated = recognizer, vocab, curated
        self.labels = recognizer.labels

    def _weight(self, gloss: str) -> float:
        s = self.vocab.get(gloss)
        return self.curated if s is not None and s.curated else 1.0

    def topk(self, frames: list[Frame], k: int = 3) -> list[tuple[str, float]]:
        scores = [(g, p * self._weight(g)) for g, p in self.r.topk(frames, 10**6)]
        z = sum(p for _, p in scores) or 1.0
        return sorted(((g, p / z) for g, p in scores), key=lambda kv: -kv[1])[:k]

    def recognize(self, seg) -> list[tuple[str, float]]:
        return self.topk(seg.context or seg.frames, 3)
