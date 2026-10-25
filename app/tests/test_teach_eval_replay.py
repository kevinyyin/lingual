import math

from server.protocol import Frame
from training.teach_eval import replay


def signing_clip(fps=15.0, rest_s=0.5, sign_s=1.2):
    """A signer at rest, raising both hands, circling them in front of the chest, and lowering them again."""
    frames, n_rest, n_sign = [], int(rest_s * fps), int(sign_s * fps)
    for i in range(2 * n_rest + n_sign):
        t = i * 1000.0 / fps
        k = i - n_rest
        up = 0 <= k < n_sign
        a = 2 * math.pi * k / max(n_sign, 1)
        wy = 0.5 + 0.05 * math.sin(a) if up else 1.0  # image y grows downward: hips at 0.95, rest below them
        wx = 0.45 + 0.05 * math.cos(a)
        pose = [[0.5, 0.6, 0.0, 1.0] for _ in range(33)]
        pose[0] = [0.5, 0.35, 0.0, 1.0]
        pose[11], pose[12] = [0.6, 0.6, 0.0, 1.0], [0.4, 0.6, 0.0, 1.0]
        pose[23], pose[24] = [0.55, 0.95, 0.0, 1.0], [0.45, 0.95, 0.0, 1.0]
        pose[13], pose[14] = [0.65, 0.75, 0.0, 1.0], [0.35, 0.75, 0.0, 1.0]
        pose[15], pose[16] = [wx + 0.1, wy, 0.0, 1.0], [wx, wy, 0.0, 1.0]
        hand = lambda cx, cy: [[cx + 0.01 * (j % 5), cy - 0.01 * (j // 5), 0.0] for j in range(21)]  # noqa: E731
        frames.append(Frame(t=t, pose=pose, left=hand(wx + 0.1, wy) if up else None, right=hand(wx, wy) if up else None))
    return frames


class Scripted:
    """A recognizer that always gives the same guesses."""

    pick_threshold = 0.5

    def __init__(self, top):
        self.top = top

    def topk(self, frames, k=3):
        return self.top[:k]

    def recognize(self, seg):
        return self.top[:3]


def session(top, practice=None):
    return replay.make_session(Scripted(top), practice=practice or {"HELLO", "NICE", "MEET", "GOODBYE"})


def test_confident_right_sign_is_spoken_straight_away():
    out = replay.translator(signing_clip(), "HELLO", session([("HELLO", 0.9), ("NICE", 0.05), ("MEET", 0.05)]))
    assert out["outcome"] == "right_direct" and out["segments"] == 1


def test_unsure_read_goes_to_the_picker_and_the_signer_picks_their_sign():
    out = replay.translator(signing_clip(), "HELLO", session([("NICE", 0.4), ("HELLO", 0.35), ("MEET", 0.25)]))
    assert out["outcome"] == "right_picked"


def test_picker_without_the_sign_is_none_of_these():
    out = replay.translator(signing_clip(), "HELLO", session([("NICE", 0.4), ("MEET", 0.35), ("GOODBYE", 0.25)]))
    assert out["outcome"] == "none_of_these"


def test_confident_wrong_sign_is_spoken():
    out = replay.translator(signing_clip(), "HELLO", session([("NICE", 0.8), ("HELLO", 0.1), ("MEET", 0.1)]))
    assert out["outcome"] == "wrong_spoken"


def test_no_hands_raised_is_no_turn():
    still = [f for f in signing_clip() if f.left is None]
    assert replay.translator(still, "HELLO", session([("HELLO", 0.9)]))["outcome"] == "no_turn"


def test_lesson_passes_when_the_sign_is_in_the_course_top_three():
    top = [("NICE", 0.5), ("MEET", 0.2), ("HELLO", 0.2), ("GOODBYE", 0.1)]
    assert replay.lesson(signing_clip(), "HELLO", session(top))["pass"]
    assert not replay.lesson(signing_clip(), "GOODBYE", session(top))["pass"]


def test_lesson_reads_among_course_signs_only():
    top = [("WATER", 0.6), ("HELLO", 0.2), ("NICE", 0.1), ("MEET", 0.1)]  # WATER is not in this course
    out = replay.lesson(signing_clip(), "HELLO", session(top))
    assert out["pass"] and all(g != "WATER" for g, _ in out["reads"][0])


def test_course_signs_come_from_the_curriculum():
    signs = replay.course_signs()
    assert {"HELLO", "THANK-YOU", "MOTHER"} <= signs and len(signs) >= 40
