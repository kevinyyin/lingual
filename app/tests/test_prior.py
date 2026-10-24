from server.prior import PriorRecognizer
from server.vocab import Sign, Vocab


class Flat:
    labels = ["HELLO", "KISSFIST"]

    def topk(self, frames, k=3):
        return [("KISSFIST", 0.6), ("HELLO", 0.4)][:k]


def test_curated_signs_outweigh_generated_ones():
    vocab = Vocab([Sign("HELLO", "hello", "social", "HELLO"), Sign("KISSFIST", "kiss fist", "noun", "KISSFIST", curated=False)])
    top = PriorRecognizer(Flat(), vocab, curated=10).topk([], 2)
    assert [g for g, _ in top] == ["HELLO", "KISSFIST"]
    assert abs(sum(p for _, p in top) - 1) < 1e-9
    assert abs(top[0][1] - 4 / 4.6) < 1e-9
