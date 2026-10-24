import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from roundtrip import score  # noqa: E402


def test_score_counts_topk():
    rows = [("A", ["A", "B", "C", "D", "E"]), ("B", ["A", "C", "B", "D", "E"]), ("C", ["A", "B", "D", "E", "F"])]
    s = score(rows)
    assert s == {"top1": 1 / 3, "top3": 2 / 3, "top5": 2 / 3, "n": 3}


def test_score_empty():
    assert score([]) == {"top1": 0.0, "top3": 0.0, "top5": 0.0, "n": 0}
