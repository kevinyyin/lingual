"""Combines sign recognizers that have different label sets (ASL Citizen ST-GCN: 2,731 signs; spike model: 143).

Each sign's score is the weighted mean probability over only the members that know that sign, so a sign one model
lacks is not dragged down by that model's silence; scores are then renormalized over the shared vocabulary.
"""

from .protocol import Frame


class EnsembleRecognizer:
    def __init__(self, members: list[tuple[object, float]], vocab):
        self.members, self.glosses = members, vocab.glosses
        self.labels = sorted({g for m, _ in members for g in m.labels if g in self.glosses})

    def topk(self, frames: list[Frame], k: int = 3) -> list[tuple[str, float]]:
        num: dict[str, float] = {}
        den: dict[str, float] = {}
        for m, w in self.members:
            probs = {g: p for g, p in m.topk(frames, 10**6) if g in self.glosses}
            total = sum(probs.values()) or 1.0
            for g in {g for g in m.labels if g in self.glosses}:
                num[g] = num.get(g, 0.0) + w * probs.get(g, 0.0) / total
                den[g] = den.get(g, 0.0) + w
        scores = {g: num[g] / den[g] for g in num}
        z = sum(scores.values()) or 1.0
        return sorted(((g, s / z) for g, s in scores.items()), key=lambda kv: -kv[1])[:k]

    def recognize(self, seg) -> list[tuple[str, float]]:
        return self.topk(seg.context or seg.frames, 3)
