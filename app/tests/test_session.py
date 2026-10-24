import json

import pytest

from frames import make_frame
from server.protocol import AvatarDoneMsg, CalibrateMsg, FrameMsg, PickMsg, PracticeMsg, SayMsg, SignPlanMsg, SignToken
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


def make_session(tmp_path, events=(), results=(), voice=None, composer=None):
    sent = []

    async def send(m):
        sent.append(m)

    s = Session(
        send=send,
        vocab=load_vocab(),
        make_segmenter=lambda b: FakeSegmenter(events),
        recognizer=FakeRecognizer(results),
        readers=FakeReaders(),
        composer=composer or FakeComposer(),
        voice=voice or FakeVoice(),
        gloss=FakeGloss(),
        log_path=tmp_path / "turns.jsonl",
    )
    return s, sent


async def calibrate(s):
    await s.on_client(CalibrateMsg())
    for i in range(60):
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
    assert s.baseline.rest_y < rest


async def test_sign_turn_speaks_and_logs(tmp_path):
    events = [TurnStart(), TurnEnd(segments=[seg(), seg()], frames=[make_frame()] * 20, rest_ms=500)]
    s, sent = make_session(tmp_path, events, [[("COFFEE", 0.9)], [("HAVE", 0.8)]])
    await calibrate(s)
    await push_frames(s, 2)
    said = [m for m in sent if m.type == "you_said"]
    assert said and said[0].text == "Do you have coffee?" and said[0].emotion == "happy"
    assert states(sent)[-3:] == ["signing", "speaking", "ready"]
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
    s, sent = make_session(tmp_path, events, [[("COFFEE", 0.9)]], voice=FakeVoice(fail=True))
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
    s, _ = make_session(tmp_path, events, [[("NAME", 0.9)]], composer=composer)
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
    s, sent = make_session(tmp_path, events, [[("NAME", 0.9)]], composer=composer)
    s.spell_reader = EmptySpell()
    await calibrate(s)
    await push_frames(s, 2)
    assert composer.calls[0][:2] == (["NAME"], [])
    assert not [m for m in sent if m.type == "pick_request"] and s.state == "ready"


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
    s, sent = make_session(tmp_path, [TurnStart(), end, TurnStart(), end], [[("COFFEE", 0.9)], [("TEA", 0.9)]], composer=composer)
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


async def test_practice_turn_sends_guesses_without_speaking(tmp_path):
    events = [TurnStart(), TurnEnd(segments=[seg()], frames=[make_frame()] * 20, rest_ms=0)]
    composer = FakeComposer()
    s, sent = make_session(tmp_path, events, [[("COFFEE", 0.3), ("TEA", 0.2)]], composer=composer)
    await s.on_client(PracticeMsg(on=True))
    await calibrate(s)
    await push_frames(s, 2)
    (rec,) = [m for m in sent if m.type == "recognized"]
    assert [[o.gloss for o in seg_] for seg_ in rec.segments] == [["COFFEE", "TEA"]]
    assert rec.segments[0][0].english == "coffee"
    assert not composer.calls and not [m for m in sent if m.type in ("pick_request", "you_said")]
    assert states(sent)[-2:] == ["signing", "ready"]


async def test_practice_reads_a_sign_cut_into_pieces_as_one(tmp_path):
    events = [TurnStart(), TurnEnd(segments=[seg(), seg()], frames=[make_frame()] * 20, rest_ms=0)]
    pieces, whole = [("DARK", 0.2)], [("HELLO", 0.6), ("NICE", 0.1)]
    s, sent = make_session(tmp_path, events, [pieces, pieces, whole])
    await s.on_client(PracticeMsg(on=True))
    await calibrate(s)
    await push_frames(s, 2)
    (rec,) = [m for m in sent if m.type == "recognized"]
    assert [[o.gloss for o in seg_] for seg_ in rec.segments] == [["HELLO", "NICE"]]


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


async def test_say_lines_never_pile_up(tmp_path):
    """One line plays; only the newest waiting line follows it (older ones are stale by then)."""
    voice = GateVoice()
    s, _ = make_session(tmp_path, voice=voice)
    await s.on_client(PracticeMsg(on=True))
    await s.on_client(SayMsg(text="one"))
    await asyncio.sleep(0.01)  # "one" is playing
    for t in ("two", "three"):
        await s.on_client(SayMsg(text=t))
    voice.release.set()
    await asyncio.sleep(0.05)
    assert voice.said == ["one", "three"]


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
