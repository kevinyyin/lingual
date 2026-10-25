import math

from frames import make_frame
from server.segmenter import CONTEXT_MS, HOLD_KEEP_MS, PAUSE_MS, REST_MS, Segmenter
from server.session import Baseline, TurnEnd, TurnStart, compute_baseline

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


def test_fingers_moving_somewhat_more_than_the_wrist_is_spell():
    # real spelling (Kaggle, 185 phrases): fingers move 1.8x the wrist (median), 1.3x at the 10th percentile
    spell = [frame(3 + i, (0.4 + 0.01 * (i % 2), 0.5), spread=0.03 + 0.015 * (i % 2)) for i in range(20)]
    end = run(rest(0, 3) + spell + rest(23, REST_FRAMES + 2))[-1]
    assert [s.kind for s in end.segments] == ["spell"]


def test_the_other_hand_seen_for_a_moment_does_not_stop_spelling():
    spell = [frame(3 + i, (0.4, 0.5), spread=0.01 + 0.05 * (i % 2)) for i in range(20)]
    for f in spell[5:7]:
        f.left = [[0.6, 0.7, 0.0]] * 21  # the resting hand drifts into the picture
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


def uneven(t, left_y=0.95, right_y=0.75):
    """Hands resting at different heights (one on the lap, one on the desk); no hand landmarks."""
    f = make_frame(t=t * 33.0, left=False, right=False)
    f.pose[15] = [0.6, left_y, 0.0, 1.0]
    f.pose[16] = [0.4, right_y, 0.0, 1.0]
    return f


def test_hands_resting_at_different_heights_do_not_start_a_turn():
    seg = Segmenter(compute_baseline([uneven(i) for i in range(60)]))
    assert [e for e in (seg.push(uneven(60 + i)) for i in range(30)) if e is not None] == []


def test_the_lower_resting_hand_counts_as_raised_against_its_own_rest():
    seg = Segmenter(compute_baseline([uneven(i, right_y=0.88) for i in range(60)]))
    up = uneven(60, left_y=0.87, right_y=0.88)
    up.left = make_frame().left  # a hand raised into view is in the hand tracker's sight
    assert isinstance(seg.push(up), TurnStart)  # 0.08 above where it rested


def test_a_hand_moving_during_calibration_does_not_lift_the_rest_line():
    calib = [frame(i, (0.4, 0.5 if i % 10 < 3 else REST_Y), hands=()) for i in range(60)]  # mid-sign for 30% of it
    seg = Segmenter(compute_baseline(calib))
    assert isinstance(seg.push(frame(60, (0.4, REST_Y - 0.1))), TurnStart)  # a sign low in front of the body


def wrist_path(frames):
    return [tuple(f.pose[16][:2]) for f in frames]


def contains_run(frames, piece):
    a, b = wrist_path(frames), wrist_path(piece)
    return any(a[i:i + len(b)] == b for i in range(len(a) - len(b) + 1))


def assert_closed_up(whole):
    """No gap where frames were cut: steps stay the frame interval."""
    assert max(b.t - a.t for a, b in zip(whole, whole[1:])) <= 33.0 + 1e-6


def test_turn_end_carries_the_whole_turn_for_reading_as_one_sign():
    # a 1.3 s hold between the pieces: longer than both contexts together, so they leave a gap the whole turn must fill
    fs = rest(0, 20) + moving(20, 12) + still(32, 40) + moving(72, 12, x0=0.45) + rest(84, REST_FRAMES + 2)
    end = run(fs)[-1]
    first, second = end.segments
    assert end.whole[0].t < end.frames[0].t  # the hands rising, from before the turn opened
    assert contains_run(end.whole, first.frames) and contains_run(end.whole, second.frames)  # both pieces, whole
    assert wrist_path(end.whole)[-1] != wrist_path(second.frames)[-1]  # and settling after the last sign
    assert_closed_up(end.whole)


def test_a_hold_in_the_whole_turn_is_cut_to_its_start():
    # a learner raises the hands, holds them in position, then signs: the sign model learned from recordings that flow
    # from rest straight into the sign, and read the held version worse (course replays, 1 s hold: 87% vs 92%)
    fs = rest(0, 3) + moving(3, 10) + still(13, 30) + moving(43, 12, x0=0.45) + rest(55, REST_FRAMES + 2)
    end = run(fs)[-1]
    held = sum(p == (0.4, 0.5) for p in wrist_path(end.whole))
    assert 1 <= held <= HOLD_KEEP_MS / 33 + 4  # of 30 still frames, the hold's start (smoothed energy falls in ~3 frames)
    assert contains_run(end.whole, fs[3:13]) and contains_run(end.whole, fs[43:55])  # the rise and the sign, whole
    assert_closed_up(end.whole)


def down(t0, n, x=0.4, y0=0.5):
    """The right wrist going from signing height back to rest over n frames."""
    return [frame(t0 + i, (x, y0 + (REST_Y - y0) * (i + 1) / n)) for i in range(n)]


def test_the_hands_going_back_down_after_the_last_signs_hold_is_not_a_sign():
    fs = rest(0, 3) + moving(3, 12) + still(15, PAUSE_FRAMES + 3) + down(22, 10) + rest(32, REST_FRAMES + 2)
    end = run(fs)[-1]
    assert len(end.segments) == 1 and end.segments[0].frames[-1].t < 22 * 33.0  # the sign, not the hands going down


def test_a_last_sign_flowing_straight_into_rest_is_kept():
    fs = (rest(0, 3) + moving(3, 12) + still(15, PAUSE_FRAMES + 3) + moving(22, 20, x0=0.45) + down(42, 10, x=0.45)
          + rest(52, REST_FRAMES + 2))
    assert len(run(fs)[-1].segments) == 2


def test_a_turns_only_piece_is_kept_even_when_it_only_goes_down():
    fs = rest(0, 3) + still(3, PAUSE_FRAMES + 3) + down(10, 10) + rest(20, REST_FRAMES + 2)  # a downward sign, raised fast
    assert len(run(fs)[-1].segments) == 1


def test_a_spelled_word_ending_the_turn_is_kept_even_if_the_wrist_drifts_down():
    spell = [frame(22 + i, (0.4, 0.5 + 0.002 * i), spread=0.01 + 0.05 * (i % 2)) for i in range(10)]
    fs = rest(0, 3) + moving(3, 12) + still(15, PAUSE_FRAMES + 3) + spell + rest(32, REST_FRAMES + 2)
    assert [s.kind for s in run(fs)[-1].segments] == ["sign", "spell"]


def offscreen(t0, n):
    """Hands below a laptop's picture: the hand tracker sees none, while the pose model keeps guessing the wrists, and
    its guesses wander (y 1.05-1.15 in the 2026-10-24 log, against a rest line learned from guesses at 1.24-1.38)."""
    out = []
    for i in range(n):
        f = frame(t0 + i, (0.4, 1.05 + 0.01 * ((i * 7) % 10)), hands=())
        f.pose[15] = [0.6, 1.1, 0.0, 0.05]
        out.append(f)
    return out


def test_hands_out_of_the_picture_end_the_turn_though_the_pose_model_guesses_them_up():
    seg = Segmenter(Baseline(rest_y=(1.38, 1.235), brow={}, motion_floor=0.0))
    evs = [e for e in (seg.push(f) for f in offscreen(0, 20) + moving(20, 14) + offscreen(34, REST_FRAMES + 2)) if e]
    assert [type(e) for e in evs] == [TurnStart, TurnEnd]  # resting off-screen starts nothing, and ends the sign's turn
    assert len(evs[-1].segments) == 1


def test_a_hand_lost_for_a_moment_mid_sign_is_a_pause_not_hands_down():
    fs = moving(3, 20)
    for f in fs[6:9]:
        f.right = None  # the tracker loses the hand for ~100 ms
    end = run(rest(0, 3) + fs + rest(23, REST_FRAMES + 2))[-1]
    assert isinstance(end, TurnEnd) and len(end.segments) == 1


def jump(d):
    """The hands raised and still, one step of `d` to the side, still again, and down: energy above the pause threshold
    for a few frames after the step (d 0.01: 1 frame, 0.05: 3)."""
    return rest(0, 5) + still(5, 5) + still(10, PAUSE_FRAMES + 3, x=0.4 + d) + rest(20 + PAUSE_FRAMES, REST_FRAMES + 2)


def test_a_turns_only_motion_shorter_than_a_segment_is_kept_from_3_frames():
    (s,) = run(jump(0.05))[-1].segments
    assert len(s.frames) == 3 and s.context


def test_a_turns_only_motion_of_1_frame_is_no_segment():
    assert run(jump(0.01))[-1].segments == []
