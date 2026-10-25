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


def whole(run):
    return [(run[0].t, run[-1].t)]


def test_touching_pieces_read_as_one_sign_inside_one_segment_are_one_sign():
    run = signing(0, 16, 0.1) + dip(16, 2, 0.1) + signing(18, 16, 0.1)  # 2.2 s: longer than one piece can be
    assert len(recut(run, run, Majority())) == 2  # no segment to merge inside
    (piece,) = recut(run, run, Majority(), whole(run))
    assert piece[1][0][0] == "S0.1" and piece[0].frames == run
    assert piece[0].context[0].t == run[0].t and piece[0].context[-1].t == run[-1].t


def test_same_sign_pieces_are_not_merged_past_2500_ms():
    run = signing(0, 19, 0.1) + dip(19, 2, 0.1) + signing(21, 19, 0.1)  # 2.6 s
    assert [top[0][0] for _, top in recut(run, run, Majority(), whole(run))] == ["S0.1", "S0.1"]


def test_one_sign_twice_with_a_pause_between_through_read_turn_stays_two(tmp_path):
    from test_session import make_session

    from server.session import Segment, TurnEnd

    a, pause, b = signing(0, 14, 0.1), dip(14, 6, 0.1), signing(20, 14, 0.1)  # a still 400 ms between
    ev = TurnEnd(segments=[Segment(frames=a, kind="sign", context=a), Segment(frames=b, kind="sign", context=b)],
                 frames=a + pause + b, rest_ms=0)
    s, _ = make_session(tmp_path, recognizer=Majority())
    assert [top[0][0] for _, top in s._read_turn(ev)] == ["S0.1", "S0.1"]
    assert recut(a + pause + b, a + pause + b, Majority(), whole(a + pause + b)) is not None  # one segment: merged
    assert len(recut(a + pause + b, a + pause + b, Majority(), whole(a + pause + b))) == 1
