import json

import pytest

from frames import make_frame
from server import plush
from server.protocol import (
    AvatarDoneMsg, CalibrateMsg, CheerMsg, FrameMsg, ModeMsg, PickMsg, PracticeMsg, RedoMsg, SayMsg, SignPlanMsg, SignToken,
)
from server.session import Baseline, Segment, Session, TurnEnd, TurnStart, compute_baseline
from server.vocab import load_vocab


class FakeSegmenter:
    """Emits scripted events, one per push."""

    def __init__(self, events):
        self.events = list(events)

    def push(self, f):
        return self.events.pop(0) if self.events else None


class FakeRecognizer:
    def __init__(self, results):
        self.results = list(results)

    def recognize(self, seg):
        return self.results.pop(0)


class FakeReaders:
    def read(self, frames, baseline):
        return {"question": "yn", "negation": False}, {"label": "happy", "intensity": 0.7}


class FakeComposer:
    def __init__(self):
        self.calls = []

    async def compose(self, signs, spelled, grammar, emotion):
        self.calls.append((signs, spelled, grammar, emotion))
        return "Do you have coffee?"


class FakeVoice:
    def __init__(self, fail=False):
        self.fail = fail

    async def speak(self, text, emotion):
        yield "first_audio"
        if self.fail:
            raise RuntimeError("socket dropped")
        yield "done"


class FakeGloss:
    async def plan(self, english):
        return SignPlanMsg(tokens=[SignToken(kind="sign", gloss="HELLO")], grammar="statement", tone="neutral", english=english)


def seg():
    return Segment(frames=[make_frame()] * 8, kind="sign")


def make_session(tmp_path, events=(), results=(), voice=None, composer=None, recognizer=None):
    sent = []

    async def send(m):
        sent.append(m)

    s = Session(
        send=send,
        vocab=load_vocab(),
        make_segmenter=lambda b: FakeSegmenter(events),
        recognizer=recognizer or FakeRecognizer(results),
        readers=FakeReaders(),
        composer=composer or FakeComposer(),
        voice=voice or FakeVoice(),
        gloss=FakeGloss(),
        log_path=tmp_path / "turns.jsonl",
    )
    return s, sent


async def calibrate(s):
    await s.on_client(CalibrateMsg())
    for i in range(62):  # 2 s at 30 fps
        await s.on_client(FrameMsg(**make_frame(t=i * 33).model_dump()))


async def push_frames(s, n):
    for i in range(n):
        await s.on_client(FrameMsg(**make_frame(t=10_000 + i * 33).model_dump()))
    await s.idle()


def states(sent):
    return [m.state for m in sent if m.type == "state"]


def log_lines(tmp_path):
    p = tmp_path / "turns.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []


async def test_calibration_sets_baseline(tmp_path):
    s, sent = make_session(tmp_path)
    await calibrate(s)
    assert s.state == "ready"
    rest = make_frame().pose[15][1]
    # Image y grows downward: the rest line sits above (smaller y than) the resting wrists.
    assert all(y < rest for y in s.baseline.rest_y)


async def test_calibration_takes_two_seconds_at_the_phones_frame_rate(tmp_path):
    s, sent = make_session(tmp_path)
    await s.on_client(CalibrateMsg())
    for i in range(20):  # 10 frames/s: 1.9 s
        await s.on_client(FrameMsg(**make_frame(t=i * 100).model_dump()))
    assert s.state == "calibrating"
    await s.on_client(FrameMsg(**make_frame(t=2000).model_dump()))
    assert s.state == "ready"


async def test_sign_turn_speaks_and_logs(tmp_path):
    events = [TurnStart(), TurnEnd(segments=[seg(), seg()], frames=[make_frame()] * 20, rest_ms=500)]
    s, sent = make_session(tmp_path, events, [[("COFFEE", 0.99)], [("HAVE", 0.98)]])
    await calibrate(s)
    await push_frames(s, 2)
    said = [m for m in sent if m.type == "you_said"]
    assert said and said[0].text == "Do you have coffee?" and said[0].emotion == "happy"
    assert states(sent)[-4:] == ["signing", "reading", "speaking", "ready"]  # "reading": hands down, the turn is read
    (line,) = log_lines(tmp_path)
    assert line["direction"] == "sign2speech"
    assert {"hands_rest", "recognized", "picked", "composed", "first_audio"} <= set(line["stamps"])
    assert line["stamps"]["recognized"] - line["stamps"]["hands_rest"] >= 500


async def test_low_confidence_requests_pick(tmp_path):
    events = [TurnStart(), TurnEnd(segments=[seg()], frames=[make_frame()] * 20, rest_ms=0)]
    composer = FakeComposer()
    s, sent = make_session(tmp_path, events, [[("COFFEE", 0.4), ("TEA", 0.3), ("MILK", 0.1)]], composer=composer)
    await calibrate(s)
    await push_frames(s, 2)
    (req,) = [m for m in sent if m.type == "pick_request"]
    assert [o.gloss for o in req.options] == ["COFFEE", "TEA", "MILK"]
    assert req.options[1].english == "tea"
    assert s.state == "picking" and not [m for m in sent if m.type == "you_said"]
    await s.on_client(PickMsg(segment=0, gloss="TEA"))
    await s.idle()
    assert composer.calls[0][0] == ["TEA"]
    assert [m for m in sent if m.type == "you_said"]


async def test_finished_speech_during_signing_wins_over_the_half_read_sign_turn(tmp_path):
    s, sent = make_session(tmp_path, [TurnStart()])  # e.g. the hearing person's own hands moving as they talk
    await calibrate(s)
    await push_frames(s, 1)
    assert s.state == "signing"
    await s.on_transcript("hello", False)  # partial words alone don't interrupt
    assert s.state == "signing"
    await s.on_transcript("hello", True)
    await s.idle()
    assert [m for m in sent if m.type == "sign_plan"]


async def test_finished_speech_closes_an_open_picker_and_is_signed(tmp_path):
    events = [TurnStart(), TurnEnd(segments=[seg()], frames=[make_frame()] * 20, rest_ms=0)]
    s, sent = make_session(tmp_path, events, [[("COFFEE", 0.4), ("TEA", 0.3), ("MILK", 0.1)]])
    await calibrate(s)
    await push_frames(s, 2)
    await asyncio.sleep(0.01)
    assert s.state == "picking"
    await s.on_transcript("do you want coffee", True)
    await s.idle()
    assert [m for m in sent if m.type == "sign_plan"]
    assert not [m for m in sent if m.type == "you_said"]  # the abandoned sign turn says nothing


async def test_signing_mode_ignores_speech(tmp_path):
    s, sent = make_session(tmp_path)
    await s.on_client(ModeMsg(mode="signing"))
    await calibrate(s)
    await s.on_transcript("hello there", True)
    await s.idle()
    assert s.state == "ready" and not [m for m in sent if m.type in ("sign_plan", "caption")]


async def test_voice_mode_ignores_the_camera_and_drops_a_sign_turn_in_flight(tmp_path):
    s, sent = make_session(tmp_path, [TurnStart(), TurnStart()])
    await calibrate(s)
    await push_frames(s, 1)
    assert s.state == "signing"
    await s.on_client(ModeMsg(mode="voice"))
    assert s.state == "ready"
    await push_frames(s, 1)  # a hand raised while talking starts no sign turn
    assert s.state == "ready"
    await s.on_transcript("hello", True)
    await s.idle()
    assert [m for m in sent if m.type == "sign_plan"]


async def test_signs_that_make_no_sentence_are_shown_as_unclear_and_not_spoken(tmp_path):
    class NoSentence(FakeComposer):
        async def compose(self, signs, spelled, grammar, emotion):
            return ""

    events = [TurnStart(), TurnEnd(segments=[seg()], frames=[make_frame()] * 20, rest_ms=0)]
    voice = RecordingVoice()
    s, sent = make_session(tmp_path, events, [[("PIZZA", 0.99)]], composer=NoSentence(), voice=voice)
    await calibrate(s)
    await push_frames(s, 2)
    (said,) = [m for m in sent if m.type == "you_said"]
    assert said.unclear and said.text == "" and not voice.said and s.state == "ready"


async def test_transcript_during_speaking_dropped(tmp_path):
    s, sent = make_session(tmp_path)
    await calibrate(s)
    s.state = "speaking"
    await s.on_transcript("hello", True)
    await s.idle()
    assert not [m for m in sent if m.type in ("sign_plan", "caption")]


async def test_empty_sign_turn_returns_ready(tmp_path):
    s, sent = make_session(tmp_path, [TurnStart(), TurnEnd(segments=[], frames=[], rest_ms=0)])
    await calibrate(s)
    await push_frames(s, 2)
    assert not [m for m in sent if m.type == "you_said"]
    assert s.state == "ready"


async def test_blank_transcript_ignored(tmp_path):
    s, sent = make_session(tmp_path)
    await calibrate(s)
    await s.on_transcript("   ", True)
    await s.idle()
    assert not [m for m in sent if m.type == "sign_plan"] and s.state == "ready"


async def test_voice_failure_still_shows_text(tmp_path):
    events = [TurnStart(), TurnEnd(segments=[seg()], frames=[make_frame()] * 20, rest_ms=0)]
    s, sent = make_session(tmp_path, events, [[("COFFEE", 0.99)]], voice=FakeVoice(fail=True))
    await calibrate(s)
    await push_frames(s, 2)
    assert [m for m in sent if m.type == "you_said"]
    assert log_lines(tmp_path)[0]["error"]
    assert s.state == "ready"


async def test_speech_turn_plans_and_waits_for_avatar(tmp_path):
    s, sent = make_session(tmp_path)
    await calibrate(s)
    await s.on_transcript("hi the", False)
    await s.on_transcript("hi there", True)
    await s.idle()
    caps = [m for m in sent if m.type == "caption"]
    assert caps[-1].final and caps[-1].text == "hi there"
    assert [m for m in sent if m.type == "sign_plan"][0].english == "hi there"
    assert s.state == "avatar"
    await s.on_client(AvatarDoneMsg())
    assert s.state == "ready"
    line = log_lines(tmp_path)[0]
    assert line["direction"] == "speech2sign"
    assert {"speech_end", "planned", "avatar_start"} <= set(line["stamps"])


async def test_speech_turn_works_before_calibration(tmp_path):
    """Speech -> sign never uses the signer's baseline: it must not wait for the camera."""
    s, sent = make_session(tmp_path)
    assert s.state == "calibrating"
    await s.on_transcript("hi there", True)
    await s.idle()
    assert [m for m in sent if m.type == "sign_plan"][0].english == "hi there"
    await s.on_client(AvatarDoneMsg())
    assert s.state == "calibrating"  # back to waiting for the camera, not "ready" without a baseline
    await calibrate(s)
    assert s.state == "ready"


def test_compute_baseline_brows():
    frames = [make_frame()] * 60
    for f in frames:
        f.blend = {"browInnerUp": 0.2, "mouthSmileLeft": 0.5}
    b = compute_baseline(frames)
    assert isinstance(b, Baseline)
    assert b.brow == pytest.approx({"browInnerUp": 0.2})


class FakeSpell:
    def recognize(self, seg):
        return [("ANA", 0.9), ("ANN", 0.05)]


async def test_spell_segment_routed_to_spell_reader(tmp_path):
    spell = Segment(frames=[make_frame()] * 8, kind="spell")
    events = [TurnStart(), TurnEnd(segments=[seg(), spell], frames=[make_frame()] * 20, rest_ms=0)]
    composer = FakeComposer()
    s, _ = make_session(tmp_path, events, [[("NAME", 0.99)]], composer=composer)
    s.spell_reader = FakeSpell()
    await calibrate(s)
    await push_frames(s, 2)
    signs, spelled, *_ = composer.calls[0]
    assert signs == ["NAME"] and spelled == ["ANA"]


class EmptySpell:
    def recognize(self, seg):
        return []


async def test_unreadable_spell_segment_is_dropped(tmp_path):
    spell = Segment(frames=[make_frame()] * 8, kind="spell")
    events = [TurnStart(), TurnEnd(segments=[seg(), spell], frames=[make_frame()] * 20, rest_ms=0)]
    composer = FakeComposer()
    s, sent = make_session(tmp_path, events, [[("NAME", 0.99)]], composer=composer)
    s.spell_reader = EmptySpell()
    await calibrate(s)
    await push_frames(s, 2)
    assert composer.calls[0][:2] == (["NAME"], [])
    assert not [m for m in sent if m.type == "pick_request"] and s.state == "ready"


async def test_a_sign_read_only_90_percent_sure_asks_the_signer(tmp_path):
    events = [TurnStart(), TurnEnd(segments=[seg()], frames=[make_frame()] * 20, rest_ms=0)]
    composer = FakeComposer()
    s, sent = make_session(tmp_path, events, [[("COFFEE", 0.9), ("TEA", 0.06), ("MILK", 0.04)]], composer=composer)
    await calibrate(s)
    await push_frames(s, 2)
    (req,) = [m for m in sent if m.type == "pick_request"]
    assert [o.gloss for o in req.options] == ["COFFEE", "TEA", "MILK"]
    assert not composer.calls and s.state == "picking"  # nothing spoken until the signer picks


async def test_log_records_pick_count(tmp_path):
    events = [TurnStart(), TurnEnd(segments=[seg()], frames=[make_frame()] * 20, rest_ms=0)]
    s, _ = make_session(tmp_path, events, [[("COFFEE", 0.4), ("TEA", 0.3), ("MILK", 0.1)]])
    await calibrate(s)
    await push_frames(s, 2)
    await s.on_client(PickMsg(segment=0, gloss="TEA"))
    await s.idle()
    assert log_lines(tmp_path)[0]["result"]["picks"] == 1


# --- final review fixes ---
import asyncio


class SlowComposer(FakeComposer):
    def __init__(self):
        super().__init__()
        self.gate = asyncio.Event()

    async def compose(self, *a):
        await self.gate.wait()
        return await super().compose(*a)


async def test_second_turn_end_while_busy_is_ignored(tmp_path):
    end = TurnEnd(segments=[seg()], frames=[make_frame()] * 20, rest_ms=0)
    composer = SlowComposer()
    s, sent = make_session(tmp_path, [TurnStart(), end, TurnStart(), end], [[("COFFEE", 0.99)], [("TEA", 0.99)]], composer=composer)
    await calibrate(s)
    for i in range(4):
        await s.on_client(FrameMsg(**make_frame(t=10_000 + i).model_dump()))
        await asyncio.sleep(0)
    composer.gate.set()
    await s.idle()
    assert len(composer.calls) == 1 and len([m for m in sent if m.type == "you_said"]) == 1


class SlowGloss(FakeGloss):
    def __init__(self):
        self.gate, self.calls = asyncio.Event(), 0

    async def plan(self, english):
        self.calls += 1
        await self.gate.wait()
        return await super().plan(english)


async def test_second_final_transcript_while_planning_is_dropped(tmp_path):
    s, sent = make_session(tmp_path)
    s.gloss = SlowGloss()
    await calibrate(s)
    await s.on_transcript("hi there", True)
    await asyncio.sleep(0)
    await s.on_transcript("and coffee", True)
    s.gloss.gate.set()
    await s.idle()
    assert s.gloss.calls == 1 and len([m for m in sent if m.type == "sign_plan"]) == 1


async def test_none_of_these_cancels_turn(tmp_path):
    events = [TurnStart(), TurnEnd(segments=[seg()], frames=[make_frame()] * 20, rest_ms=0)]
    composer = FakeComposer()
    s, sent = make_session(tmp_path, events, [[("COFFEE", 0.4), ("TEA", 0.3), ("MILK", 0.1)]], composer=composer)
    await calibrate(s)
    await push_frames(s, 2)
    await s.on_client(PickMsg(segment=0, gloss=""))
    await s.idle()
    assert composer.calls == [] and s.state == "ready"
    assert log_lines(tmp_path)[0]["error"] == "cancelled by signer"


async def test_pick_not_offered_is_ignored(tmp_path):
    events = [TurnStart(), TurnEnd(segments=[seg()], frames=[make_frame()] * 20, rest_ms=0)]
    s, _ = make_session(tmp_path, events, [[("COFFEE", 0.4), ("TEA", 0.3), ("MILK", 0.1)]])
    await calibrate(s)
    await push_frames(s, 2)
    await s.on_client(PickMsg(segment=0, gloss="ZEBRA"))
    await asyncio.sleep(0)
    assert s.state == "picking"


async def test_recalibrate_during_picking_ends_turn(tmp_path):
    events = [TurnStart(), TurnEnd(segments=[seg()], frames=[make_frame()] * 20, rest_ms=0)]
    s, _ = make_session(tmp_path, events, [[("COFFEE", 0.4), ("TEA", 0.3), ("MILK", 0.1)]])
    await calibrate(s)
    await push_frames(s, 2)
    await s.on_client(CalibrateMsg())
    await asyncio.sleep(0.01)
    assert s._task.done() and s.state == "calibrating"


async def test_signer_leaving_frame_mid_turn_times_out(tmp_path, monkeypatch):
    import server.session as S

    monkeypatch.setattr(S, "FRAME_TIMEOUT_S", 0.05)
    s, sent = make_session(tmp_path, [TurnStart()])
    await calibrate(s)
    await push_frames(s, 1)
    assert s.state == "signing"
    await asyncio.sleep(0.3)
    assert s.state == "ready"


def test_demo_recognizer_follows_script_one_sentence_per_turn():
    from server.session import DemoRecognizer
    r = DemoRecognizer([["I", "WANT", "COFFEE"], ["YOU", "NAME", "WHAT"]])
    seg = lambda t: Segment(frames=[make_frame(t=t), make_frame(t=t + 200)], kind="sign")
    first = [r.recognize(seg(t))[0][0] for t in (0, 400, 800)]  # short pauses: one turn
    second = [r.recognize(seg(t))[0][0] for t in (5000, 5400)]  # long gap: the next sentence
    assert first == ["I", "WANT", "COFFEE"] and second == ["YOU", "NAME"]
    assert r.recognize(seg(9000))[0][0] == "I"  # wraps around the script


class WholeTurnRecognizer:
    """The full distribution for any frames, and every frame list it was asked about."""

    def __init__(self, dist):
        self.dist, self.asked = dist, []

    def topk(self, frames, k=3):
        self.asked.append(frames)
        return self.dist[:k]


async def test_practice_reads_the_whole_turn_as_one_sign_among_the_course_signs(tmp_path):
    whole = [make_frame(t=i * 66) for i in range(30)]
    events = [TurnStart(), TurnEnd(segments=[seg(), seg()], frames=whole[5:25], rest_ms=0, whole=whole)]  # sign + hands down
    rec = WholeTurnRecognizer([("HAIRCLIPPER", 0.5), ("GOODBYE", 0.2), ("SPACESHIP", 0.15), ("HELLO", 0.1), ("THANK-YOU", 0.05)])
    composer = FakeComposer()
    s, sent = make_session(tmp_path, events, recognizer=rec, composer=composer)
    await s.on_client(PracticeMsg(on=True, signs=["HELLO", "GOODBYE", "THANK-YOU"]))
    await calibrate(s)
    await push_frames(s, 2)
    (msg,) = [m for m in sent if m.type == "recognized"]
    guesses, latest = msg.segments  # one sign per exercise however the segmenter cut it, and its last 1.5 s alone
    assert [o.gloss for o in guesses] == ["GOODBYE", "HELLO", "THANK-YOU"] == [o.gloss for o in latest]
    assert guesses[0].english == "goodbye" and abs(sum(o.p for o in guesses) - 1) < 1e-6
    assert rec.asked == [whole, [f for f in whole if f.t >= whole[-1].t - 1500]]
    assert not composer.calls and not [m for m in sent if m.type in ("pick_request", "you_said")]
    assert states(sent)[-3:] == ["signing", "reading", "ready"]


async def test_practice_without_a_course_list_reads_among_every_sign(tmp_path):
    events = [TurnStart(), TurnEnd(segments=[seg()], frames=[make_frame()] * 20, rest_ms=0)]
    rec = WholeTurnRecognizer([("COFFEE", 0.6), ("TEA", 0.4)])
    s, sent = make_session(tmp_path, events, recognizer=rec)
    await s.on_client(PracticeMsg(on=True))
    await calibrate(s)
    await push_frames(s, 2)
    (msg,) = [m for m in sent if m.type == "recognized"]
    assert [o.gloss for o in msg.segments[0]] == ["COFFEE", "TEA"]
    assert rec.asked == [[make_frame()] * 20]  # no whole span from the segmenter: the turn's own frames


async def test_practice_reads_a_corrected_sign_on_its_own_too(tmp_path):
    """A wrong sign, then the right one without resting between: the whole turn reads as neither, its end as the right one."""
    class Corrected:
        def topk(self, frames, k=3):
            return [("DARK", 0.5), ("NICE", 0.3)] if frames[-1].t - frames[0].t > 1500 else [("HELLO", 0.7), ("NICE", 0.2)]

    whole = [make_frame(t=i * 66) for i in range(50)]  # 3.2 s: the wrong sign, a beat, the right one
    events = [TurnStart(), TurnEnd(segments=[seg()], frames=whole, rest_ms=0, whole=whole)]
    s, sent = make_session(tmp_path, events, recognizer=Corrected())
    await s.on_client(PracticeMsg(on=True))
    await calibrate(s)
    await push_frames(s, 2)
    (rec,) = [m for m in sent if m.type == "recognized"]
    assert [[o.gloss for o in seg_] for seg_ in rec.segments] == [["DARK", "NICE"], ["HELLO", "NICE"]]


async def test_redo_drops_the_attempt_being_read(tmp_path):
    """The learner tapped the frozen camera: the attempt still being read is dropped unread, and they sign again."""
    import time

    class Slow:
        def topk(self, frames, k=3):
            time.sleep(0.2)  # reading the attempt
            return [("HELLO", 0.9)]

    events = [TurnStart(), TurnEnd(segments=[seg()], frames=[make_frame()] * 20, rest_ms=0)]
    s, sent = make_session(tmp_path, events, recognizer=Slow())
    await s.on_client(PracticeMsg(on=True))
    await calibrate(s)
    for i in range(2):
        await s.on_client(FrameMsg(**make_frame(t=10_000 + i * 33).model_dump()))
    assert s.state == "reading"
    await s.on_client(RedoMsg())
    await asyncio.sleep(0.4)
    assert s.state == "ready" and not [m for m in sent if m.type == "recognized"]


async def test_practice_mode_ignores_speech(tmp_path):
    s, sent = make_session(tmp_path)
    await s.on_client(PracticeMsg(on=True))
    await calibrate(s)
    await s.on_transcript("hello there", True)
    await s.idle()
    assert s.state == "ready" and not [m for m in sent if m.type == "sign_plan"]


class RecordingVoice:
    def __init__(self):
        self.said = []

    async def speak(self, text, emotion):
        self.said.append((text, emotion))
        yield "done"


async def test_say_speaks_in_practice(tmp_path):
    voice = RecordingVoice()
    s, _ = make_session(tmp_path, voice=voice)
    await s.on_client(PracticeMsg(on=True))
    before = s.state
    await s.on_client(SayMsg(text="Nice!"))
    await asyncio.sleep(0.01)  # let the task run
    assert voice.said == [("Nice!", {"label": "happy", "intensity": 0.4})]
    assert s.state == before


async def test_cheer_flaps_the_plushie_in_practice_only(tmp_path, monkeypatch):
    cheers = []
    monkeypatch.setattr(plush, "cheer", lambda: cheers.append(1))
    s, _ = make_session(tmp_path)
    await s.on_client(CheerMsg())
    await s.on_client(PracticeMsg(on=True))
    await s.on_client(CheerMsg())
    assert cheers == [1]


async def test_say_rejected_outside_practice(tmp_path):
    voice = RecordingVoice()
    s, _ = make_session(tmp_path, voice=voice)
    await s.on_client(SayMsg(text="hi"))
    await asyncio.sleep(0.01)
    assert voice.said == []


class GateVoice:
    """Each line plays until released."""
    def __init__(self):
        self.said, self.release = [], asyncio.Event()

    async def speak(self, text, emotion):
        self.said.append(text)
        await self.release.wait()
        yield "done"


async def test_a_newer_line_cuts_off_the_one_still_playing(tmp_path):
    """The lesson moved on, so a line still playing (or waiting) is stale: the newest starts now, not after it."""
    voice = GateVoice()  # each line plays until released: "one" never is
    s, _ = make_session(tmp_path, voice=voice)
    await s.on_client(PracticeMsg(on=True))
    await s.on_client(SayMsg(text="one"))
    await asyncio.sleep(0.01)  # "one" is playing
    for t in ("two", "three"):
        await s.on_client(SayMsg(text=t))
    await asyncio.sleep(0.05)
    assert voice.said == ["one", "three"]  # "two" was stale before it began; "three" did not wait for "one"


class SlowVoice:
    def __init__(self):
        self.started = 0

    async def speak(self, text, emotion):
        self.started += 1
        await asyncio.sleep(10)
        yield "done"


async def test_close_cancels_queued_lines(tmp_path):
    voice = SlowVoice()
    s, _ = make_session(tmp_path, voice=voice)
    await s.on_client(PracticeMsg(on=True))
    for t in ("one", "two"):
        await s.on_client(SayMsg(text=t))
    await asyncio.sleep(0.01)
    await s.close()
    await asyncio.sleep(0.01)
    assert voice.started == 1 and not s._says


async def test_close_cancels_the_turn_in_flight(tmp_path):
    s, _ = make_session(tmp_path)
    s._task = asyncio.create_task(asyncio.sleep(10))  # a turn still composing or speaking when the page goes away
    await s.close()
    assert s._task.cancelled()
