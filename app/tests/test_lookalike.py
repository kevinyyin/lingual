import math

from frames import make_frame
from server.lookalike import LookalikeRecognizer, palm_spread, settle


def held(n=12):
    return [make_frame(t=i * 66.0) for i in range(n)]


def circling(n=12):  # the palm sways 0.03 either way (shoulders 0.2 wide): PLEASE's rub
    return [make_frame(t=i * 66.0, jitter=0.03 * math.cos(i)) for i in range(n)]


def test_a_held_palm_barely_spreads_and_a_rubbing_one_does():
    assert palm_spread(held()) < 0.01
    assert palm_spread(circling()) > 0.07
    assert palm_spread(held(3)) is None  # too few frames to tell


def test_my_and_please_are_settled_by_the_palms_circling():
    top = [("PLEASE", 0.9), ("MY", 0.06), ("FEEL", 0.01)]
    assert settle(held(), top) == [("MY", 0.9), ("PLEASE", 0.06), ("FEEL", 0.01)]
    assert settle(circling(), top) == top
    assert settle(circling(), [("MY", 0.7), ("PLEASE", 0.2)]) == [("PLEASE", 0.7), ("MY", 0.2)]


def test_other_guesses_are_left_alone():
    top = [("NAME", 0.9), ("MY", 0.05), ("PLEASE", 0.01)]  # MY and PLEASE in the top 3, but not first
    assert settle(held(), top) == top
    assert settle(held(), [("PLEASE", 0.9), ("FEEL", 0.05)]) == [("PLEASE", 0.9), ("FEEL", 0.05)]


def test_the_wrapper_settles_and_passes_everything_else_through():
    class Inner:
        labels = ["MY", "PLEASE"]
        pick_threshold = 0.5

        def topk(self, frames, k=3):
            return [("PLEASE", 0.8), ("MY", 0.1)]

    r = LookalikeRecognizer(Inner())
    assert r.topk(held())[0][0] == "MY"
    assert r.labels == ["MY", "PLEASE"] and r.pick_threshold == 0.5
