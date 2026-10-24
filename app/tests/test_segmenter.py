import math

from frames import make_frame
from server.segmenter import CONTEXT_MS, PAUSE_MS, REST_MS, Segmenter
from server.session import Baseline, TurnEnd, TurnStart

BASE = Baseline(rest_y=0.8, brow={}, motion_floor=0.0)
REST_Y = 0.95  # wrists below the rest line
# the helpers below space frames 33 ms apart (30 fps): the segmenter's ms thresholds as frame counts
REST_FRAMES = math.ceil(REST_MS / 33) + 1
PAUSE_FRAMES = math.ceil(PAUSE_MS / 33) + 1


def frame(t, wrist, spread=0.03, hands=("right",)):
    """Right wrist at `wrist` (x, y); left wrist resting. Fingertips spread `spread` from the wrist."""
    f = make_frame(t=t * 33.0, left=False, right=False)
    f.pose[15] = [0.6, REST_Y, 0.0, 1.0]
    f.pose[16] = [wrist[0], wrist[1], 0.0, 1.0]
    if "right" in hands:
        pts = [[wrist[0], wrist[1], 0.0]] * 21
        for i, tip in enumerate((4, 8, 12, 16, 20)):
            a = i * 0.6
            pts[tip] = [wrist[0] + spread * math.cos(a), wrist[1] - spread * math.sin(a), 0.0]
        f.right = pts
    return f


def run(frames):
    seg = Segmenter(BASE)
    return [e for e in (seg.push(f) for f in frames) if e is not None]


def rest(t0, n):
    return [frame(t0 + i, (0.4, REST_Y), hands=()) for i in range(n)]


def moving(t0, n, x0=0.4, y=0.5):
    return [frame(t0 + i, (x0 + 0.02 * math.sin(i), y + 0.02 * math.cos(i))) for i in range(n)]


def still(t0, n, x=0.4, y=0.5):
    return [frame(t0 + i, (x, y)) for i in range(n)]


def test_two_signs_with_pause_make_two_segments():
    fs = rest(0, 5) + moving(5, 12) + still(17, PAUSE_FRAMES + 3) + moving(30, 12, x0=0.45) + rest(42, REST_FRAMES + 2)
    events = run(fs)
    assert isinstance(events[0], TurnStart)
    end = events[-1]
    assert isinstance(end, TurnEnd)
    assert len(end.segments) == 2 and all(s.kind == "sign" for s in end.segments)


def test_hands_rest_ends_turn():
    fs = rest(0, 3) + moving(3, 12)
    seg = Segmenter(BASE)
    for f in fs:
        seg.push(f)
    for i, f in enumerate(rest(15, REST_FRAMES + 5)):
        if isinstance(seg.push(f), TurnEnd):
            assert i + 1 <= REST_FRAMES + 1
            return
    raise AssertionError("no TurnEnd")


def test_turn_end_reports_rest_time():
    end = run(rest(0, 3) + moving(3, 12) + rest(15, REST_FRAMES + 2))[-1]
    assert end.rest_ms > 0


def test_hands_never_raised_no_turn():
    assert run(rest(0, 100)) == []


def test_raise_and_drop_without_motion_gives_empty_segments():
    end = run(rest(0, 3) + still(3, 20) + rest(23, REST_FRAMES + 2))[-1]
    assert isinstance(end, TurnEnd) and end.segments == []


def test_still_wrist_busy_fingers_is_spell():
    spell = [frame(3 + i, (0.4, 0.5), spread=0.01 + 0.05 * (i % 2)) for i in range(20)]
    end = run(rest(0, 3) + spell + rest(23, REST_FRAMES + 2))[-1]
    assert [s.kind for s in end.segments] == ["spell"]


def test_hand_drops_out_mid_sign_no_crash():
    fs = moving(3, 14)
    for f in fs[5:10]:
        f.right = None
    end = run(rest(0, 3) + fs + rest(17, REST_FRAMES + 2))[-1]
    assert len(end.segments) == 1


def test_letters_split_by_holds_merge_into_one_spell():
    burst = lambda t0: [frame(t0 + i, (0.4, 0.5), spread=0.01 + 0.05 * (i % 2)) for i in range(10)]
    hold = [frame(13 + i, (0.4, 0.5), spread=0.01) for i in range(PAUSE_FRAMES + 10)]
    end = run(rest(0, 3) + burst(3) + hold + burst(30) + rest(40, REST_FRAMES + 2))[-1]
    assert [s.kind for s in end.segments] == ["spell"]


def test_short_sign_ending_at_hands_down_is_kept():
    end = run(rest(0, 3) + moving(3, 8) + rest(11, REST_FRAMES + 2))[-1]
    assert len(end.segments) == 1 and len(end.segments[0].frames) >= 7


def test_sign_context_reaches_back_before_the_turn_and_stops_halfway_to_the_next_sign():
    fs = rest(0, 20) + moving(20, 12) + still(32, PAUSE_FRAMES + 3) + moving(45, 12, x0=0.45) + rest(57, REST_FRAMES + 2)
    end = run(fs)[-1]
    first, second = end.segments
    turn_start = end.frames[0].t
    assert first.context[0].t < turn_start  # the hands rising from rest, from before the turn opened
    assert first.context[0].t >= first.frames[0].t - CONTEXT_MS
    halfway = (first.frames[-1].t + second.frames[0].t) / 2
    assert first.context[-1].t <= halfway < second.context[0].t
    assert second.context[-1].t > second.frames[-1].t  # and the hands settling after it


def test_turn_ends_after_the_same_rest_time_at_the_browsers_15_fps():
    seg = Segmenter(BASE)
    fs = [frame(i * 2, (0.4 + 0.02 * math.sin(i), 0.5)) for i in range(12)]  # 66 ms apart
    for f in fs:
        seg.push(f)
    for i in range(20):
        if isinstance(seg.push(frame(24 + 2 * i, (0.4, REST_Y), hands=())), TurnEnd):
            assert (i + 1) <= REST_FRAMES // 2 + 1  # ~half the frames, the same ~0.45 s
            return
    raise AssertionError("no TurnEnd")
