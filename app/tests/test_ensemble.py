from types import SimpleNamespace

from server.ensemble import EnsembleRecognizer


class Fixed:
    def __init__(self, probs):
        self.p, self.labels = probs, list(probs)

    def topk(self, frames, k=3):
        return sorted(self.p.items(), key=lambda kv: -kv[1])[:k]


VOCAB = SimpleNamespace(glosses={"COFFEE", "TEA", "MILK"})


def test_sign_known_by_one_member_is_not_penalized():
    a = Fixed({"COFFEE": 0.2, "TEA": 0.1, "MILK": 0.7})  # knows everything
    b = Fixed({"COFFEE": 0.6, "TEA": 0.4})  # never heard of MILK
    top = EnsembleRecognizer([(a, 1.0), (b, 1.0)], VOCAB).topk([], 3)
    assert top[0][0] == "MILK"  # b's silence about MILK must not sink it
    assert abs(sum(p for _, p in top) - 1.0) < 1e-6


def test_out_of_vocabulary_labels_are_dropped_and_weights_matter():
    a = Fixed({"COFFEE": 0.5, "TEA": 0.1, "BANANA": 0.4})
    b = Fixed({"COFFEE": 0.1, "TEA": 0.9})
    e = EnsembleRecognizer([(a, 3.0), (b, 1.0)], VOCAB)
    assert "BANANA" not in e.labels and "BANANA" not in dict(e.topk([], 5))
    assert e.topk([], 1)[0][0] == "COFFEE"  # a's renormalized 0.83 x3 outweighs b
