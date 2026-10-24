import math

from frames import make_frame
from server.grammar import read_grammar
from server.session import Baseline

BASE = Baseline(rest_y=0.8, brow={}, motion_floor=0.0)


def turn(n=40, blend=None, yaw=lambda i: 0.0, face=True):
    out = []
    for i in range(n):
        f = make_frame(t=i * 33.0)
        if face:
            f.blend = dict(blend or {})
            f.head = (yaw(i), 0.0, 0.0)
        out.append(f)
    return out


def test_raised_brows_yn():
    g = read_grammar(turn(blend={"browInnerUp": 0.5, "browOuterUpLeft": 0.4, "browOuterUpRight": 0.4}), BASE)
    assert g == {"question": "yn", "negation": False}


def test_lowered_brows_wh():
    g = read_grammar(turn(blend={"browDownLeft": 0.4, "browDownRight": 0.4}), BASE)
    assert g["question"] == "wh"


def test_headshake_negation():
    # ±10° at ~2.5 Hz over 40 frames (1.3 s): several direction changes
    g = read_grammar(turn(yaw=lambda i: 10 * math.sin(2 * math.pi * 2.5 * i / 30)), BASE)
    assert g == {"question": None, "negation": True}


def test_slow_head_turn_not_negation():
    g = read_grammar(turn(yaw=lambda i: 20 * i / 39), BASE)
    assert g["negation"] is False


def test_small_jitter_not_negation():
    g = read_grammar(turn(yaw=lambda i: 2 * (-1) ** i), BASE)
    assert g["negation"] is False


def test_baseline_subtracted():
    b = Baseline(rest_y=0.8, brow={"browInnerUp": 0.2, "browOuterUpLeft": 0.2, "browOuterUpRight": 0.2}, motion_floor=0.0)
    g = read_grammar(turn(blend={"browInnerUp": 0.25, "browOuterUpLeft": 0.25, "browOuterUpRight": 0.25}), b)
    assert g["question"] is None


def test_no_face_defaults_statement():
    assert read_grammar(turn(face=False), BASE) == {"question": None, "negation": False}


def test_brows_only_in_lead_in_ignored():
    fs = turn(blend={})
    for f in fs[:3]:  # first 7% of the turn: outside the middle 80%
        f.blend = {"browInnerUp": 1.0, "browOuterUpLeft": 1.0, "browOuterUpRight": 1.0}
    assert read_grammar(fs, BASE)["question"] is None
