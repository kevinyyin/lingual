import math

from frames import make_frame
from server.recut import recut


def signing(t0, n, label, speed=0.03):
    """n frames of one sign, the right wrist circling; `label` rides in the nose x (the fake model reads it)."""
    out = []
    for i in range(n):
        f = make_frame(t=(t0 + i) * 66.7)
        a = i * 1.1
        f.pose[16] = [0.4 + speed * math.cos(a) * i, 0.5 + speed * math.sin(a), 0.0, 1.0]
        f.pose[0][0] = label
        out.append(f)
    return out


def dip(t0, n, label):
    """The hands nearly still: where one sign ends and the next begins."""
    return [f for f in signing(t0, n, label, speed=0.001)]


class Majority:
    """Fake sign model: the majority label of the frames, unsure unless nearly all of them agree."""

    def topk(self, frames, k=3):
        labels = [round(f.pose[0][0], 2) for f in frames]
        best = max(set(labels), key=labels.count)
        return [(f"S{best}", (labels.count(best) / len(labels)) ** 4)][:k]


def test_two_signs_flowing_without_a_pause_are_cut_where_the_hands_slow():
    run = signing(0, 16, 0.1) + dip(16, 2, 0.1) + signing(18, 16, 0.2)  # 2.3 s: longer than one sign can be
    pieces = recut(run, run, Majority())
    assert [top[0][0] for _, top in pieces] == ["S0.1", "S0.2"]
    assert 16 <= round(pieces[1][0].frames[0].t / 66.7) <= 18  # at the dip
    assert all(s.kind == "sign" and s.context for s, _ in pieces)


def test_a_sign_that_slows_midway_stays_one_sign():
    run = signing(0, 10, 0.1) + dip(10, 2, 0.1) + signing(12, 10, 0.1)
    assert [top[0][0] for _, top in recut(run, run, Majority())] == ["S0.1"]


def test_where_the_hands_never_slow_any_frame_may_be_a_cut_if_asked():
    """A webcam at ~7 frames a second: 3.4 s of two signs with the wrist speeding up throughout, no dip to cut at."""
    run = []
    for i in range(24):
        f = make_frame(t=i * 143.0)
        f.pose[16] = [0.4 + 0.001 * i * i, 0.5, 0.0, 1.0]
        f.pose[0][0] = 0.1 if i < 12 else 0.2
        run.append(f)
    assert recut(run, run, Majority()) is None  # one piece would be longer than a sign can be
    assert [top[0][0] for _, top in recut(run, run, Majority(), fill_gaps=True)] == ["S0.1", "S0.2"]


def test_too_short_to_cut_is_left_to_the_segmenter():
    run = signing(0, 3, 0.1)
    assert recut(run, run, Majority()) is None
