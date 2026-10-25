import json

import pytest

from frames import make_frame
from server import session as S
from server.protocol import PickMsg
from server.session import Segment, TurnEnd, TurnStart
from test_session import calibrate, log_lines, make_session, push_frames

TABLE = {"GOOD": ["open_b", "Head"], "THANK-YOU": ["open_b", "Head"], "MINUTE": ["1", "Hand"], "HOUR": ["1", "Hand"],
         "WHERE": ["1", "Neutral"], "WHO": ["1", "Head"], "NEED": ["x", "Neutral"], "COFFEE": ["s", "Neutral"]}


class LabelledRecognizer:
    """A sign model with labels (so the look-alike table applies), scripted reads and a scripted full top-k."""

    pick_threshold = 0.5
    labels = list(TABLE)

    def __init__(self, results=(), full=()):
        self.results, self.full, self.topk_calls = list(results), list(full), 0

    def recognize(self, seg):
        return self.results.pop(0)

    def topk(self, frames, k):
        self.topk_calls += 1
        return self.full[:k]


@pytest.fixture
def table(tmp_path, monkeypatch):
    p = tmp_path / "lookalikes.json"
    p.write_text(json.dumps(TABLE), encoding="utf-8")
    monkeypatch.setattr(S, "LOOKALIKES", p)
    return p


def seg():
    return Segment(frames=[make_frame()] * 8, kind="sign")


def session(tmp_path, rec):
    return make_session(tmp_path, recognizer=rec)[0]


def test_a_sure_read_with_its_lookalike_close_behind_goes_to_the_picker(tmp_path, table):
    s = session(tmp_path, LabelledRecognizer())
    t = [("THANK-YOU", 0.6), ("GOOD", 0.3), ("COFFEE", 0.05)]
    assert s._needs_pick(t, seg()) == ("lookalike", t)
    assert s._needs_pick([("THANK-YOU", 0.6), ("GOOD", 0.1), ("COFFEE", 0.05)], seg())[0] is None  # < 0.25 x p1


def test_a_read_at_the_cap_is_spoken(tmp_path, table):
    s = session(tmp_path, LabelledRecognizer())
    assert s._needs_pick([("THANK-YOU", 0.95), ("GOOD", 0.04), ("COFFEE", 0.01)], seg())[0] is None


def test_a_picker_without_the_lookalike_gets_it_in_third_place(tmp_path, table):
    rec = LabelledRecognizer(full=[("HOUR", 0.23), ("WHERE", 0.2), ("NEED", 0.1), ("MINUTE", 0.08), ("GOOD", 0.05)])
    s = session(tmp_path, rec)
    reason, t = s._needs_pick([("HOUR", 0.23), ("WHERE", 0.2), ("NEED", 0.1)], seg())
    assert reason == "unsure+swap" and t == [("HOUR", 0.23), ("WHERE", 0.2), ("MINUTE", 0.08)]


def test_a_far_lookalike_does_not_push_out_the_third_option(tmp_path, table):
    rec = LabelledRecognizer(full=[("HOUR", 0.37), ("WHO", 0.2), ("NEED", 0.06), ("MINUTE", 0.02)])
    s = session(tmp_path, rec)
    t = [("HOUR", 0.37), ("WHO", 0.2), ("NEED", 0.06)]
    assert s._needs_pick(t, seg()) == ("unsure", t)  # MINUTE 0.02 < 0.5 x 0.06


def test_no_table_no_change(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "LOOKALIKES", tmp_path / "missing.json")
    rec = LabelledRecognizer(full=[("HOUR", 0.23), ("MINUTE", 0.2)])
    s = session(tmp_path, rec)
    assert s._alike == {}
    assert s._needs_pick([("THANK-YOU", 0.6), ("GOOD", 0.3), ("COFFEE", 0.05)], seg())[0] is None
    t = [("HOUR", 0.23), ("WHERE", 0.2), ("NEED", 0.1)]
    assert s._needs_pick(t, seg()) == ("unsure", t) and rec.topk_calls == 0


def test_the_committed_table_pairs_good_with_thank_you():
    alike = S.load_lookalikes()
    assert alike["GOOD"] == alike["THANK-YOU"] and alike["MINUTE"] == alike["HOUR"]


async def test_the_guard_offers_both_signs_and_logs_why(tmp_path, table):
    rec = LabelledRecognizer(results=[[("THANK-YOU", 0.6), ("GOOD", 0.3), ("COFFEE", 0.05)]])
    events = [TurnStart(), TurnEnd(segments=[seg()], frames=[make_frame()] * 20, rest_ms=0)]
    s, sent = make_session(tmp_path, events, recognizer=rec)
    await calibrate(s)
    await push_frames(s, 2)
    (req,) = [m for m in sent if m.type == "pick_request"]
    assert [o.gloss for o in req.options] == ["THANK-YOU", "GOOD", "COFFEE"]
    await s.on_client(PickMsg(segment=0, gloss="GOOD"))
    await s.idle()
    (line,) = log_lines(tmp_path)
    assert line["result"]["offered"] == [{"segment": 0, "reason": "lookalike", "lookalike": True,
                                          "options": [["THANK-YOU", 0.6], ["GOOD", 0.3], ["COFFEE", 0.05]]}]
