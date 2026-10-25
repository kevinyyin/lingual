"""Sign by sign (ModeMsg.by_sign): each turn is one word, shown as it is read; the words are said on Done."""
import asyncio

from frames import make_frame
from server import session as session_mod
from server.protocol import ModeMsg, PickMsg, WordsMsg
from server.session import TurnEnd, TurnStart
from test_session import FakeComposer, calibrate, make_session, push_frames, seg, states


def word_turn():
    return [TurnStart(), TurnEnd(segments=[seg(), seg()], frames=[make_frame()] * 20, rest_ms=450)]


def chips(sent):
    return [[w.gloss for w in m.words] for m in sent if m.type == "words_update"]


async def by_sign_session(tmp_path, results, turns=2, composer=None):
    s, sent = make_session(tmp_path, [e for _ in range(turns) for e in word_turn()], results, composer=composer)
    await calibrate(s)
    await s.on_client(ModeMsg(mode="signing", by_sign=True))
    return s, sent


async def test_each_turn_is_one_word_read_whole_and_done_says_them(tmp_path, monkeypatch):
    monkeypatch.setattr(session_mod, "WORDS_DONE_MS", 0.0)  # only on Done
    composer = FakeComposer()
    s, sent = await by_sign_session(tmp_path, [[("COFFEE", 0.99)], [("WANT", 0.97)]], composer=composer)
    await push_frames(s, 2)  # two segments in the turn, still one word: no cutting in this mode
    await push_frames(s, 2)
    assert chips(sent) == [["COFFEE"], ["COFFEE", "WANT"]]
    assert not [m for m in sent if m.type == "you_said"] and s.state == "ready"
    await s.on_client(WordsMsg(action="done"))
    await s.idle()
    assert composer.calls[0][0] == ["COFFEE", "WANT"]
    (said,) = [m for m in sent if m.type == "you_said"]
    assert [t.gloss for t in said.read] == ["COFFEE", "WANT"]
    assert chips(sent)[-1] == [] and states(sent)[-2:] == ["speaking", "ready"]


async def test_none_of_these_drops_only_that_word(tmp_path, monkeypatch):
    monkeypatch.setattr(session_mod, "WORDS_DONE_MS", 0.0)
    s, sent = await by_sign_session(tmp_path, [[("COFFEE", 0.99)], [("TEA", 0.3), ("MILK", 0.2), ("WATER", 0.1)]])
    await push_frames(s, 2)
    await push_frames(s, 2)
    (req,) = [m for m in sent if m.type == "pick_request"]
    assert req.segment == 1 and s.state == "picking"
    await s.on_client(PickMsg(segment=1, gloss=""))
    await s.idle()
    assert s.state == "ready" and s._words and [w.gloss for w, _ in s._words] == ["COFFEE"]


async def test_a_picked_word_is_added(tmp_path, monkeypatch):
    monkeypatch.setattr(session_mod, "WORDS_DONE_MS", 0.0)
    s, sent = await by_sign_session(tmp_path, [[("TEA", 0.3), ("MILK", 0.2), ("WATER", 0.1)]], turns=1)
    await push_frames(s, 2)
    await s.on_client(PickMsg(segment=0, gloss="MILK"))
    await s.idle()
    assert chips(sent) == [["MILK"]]


async def test_undo_drops_the_last_word(tmp_path, monkeypatch):
    monkeypatch.setattr(session_mod, "WORDS_DONE_MS", 0.0)
    s, sent = await by_sign_session(tmp_path, [[("COFFEE", 0.99)], [("WANT", 0.97)]])
    await push_frames(s, 2)
    await push_frames(s, 2)
    await s.on_client(WordsMsg(action="undo"))
    assert chips(sent)[-1] == ["COFFEE"]


async def test_words_are_said_after_a_pause_unless_another_sign_starts(tmp_path, monkeypatch):
    monkeypatch.setattr(session_mod, "WORDS_DONE_MS", 50.0)
    composer = FakeComposer()
    s, sent = await by_sign_session(tmp_path, [[("COFFEE", 0.99)], [("WANT", 0.97)]], composer=composer)
    await push_frames(s, 2)
    await push_frames(s, 1)  # the next sign starts before the pause is up: nothing said yet
    await asyncio.sleep(0.1)
    assert not composer.calls
    await push_frames(s, 1)  # its word is read; then the pause
    await asyncio.sleep(0.2)
    await s.idle()
    assert composer.calls and composer.calls[0][0] == ["COFFEE", "WANT"]


async def test_switching_to_voice_clears_the_words(tmp_path, monkeypatch):
    monkeypatch.setattr(session_mod, "WORDS_DONE_MS", 0.0)
    s, sent = await by_sign_session(tmp_path, [[("COFFEE", 0.99)]], turns=1)
    await push_frames(s, 2)
    await s.on_client(ModeMsg(mode="voice"))
    assert not s._words and chips(sent)[-1] == []


async def test_without_by_sign_a_turn_is_still_a_whole_sentence(tmp_path):
    composer = FakeComposer()
    s, sent = make_session(tmp_path, word_turn(), [[("COFFEE", 0.99)], [("HAVE", 0.98)]], composer=composer)
    await calibrate(s)
    await s.on_client(ModeMsg(mode="signing"))
    await push_frames(s, 2)
    assert composer.calls[0][0] == ["COFFEE", "HAVE"] and not chips(sent)


async def test_words_are_still_said_after_a_turn_that_added_no_word(tmp_path, monkeypatch):
    monkeypatch.setattr(session_mod, "WORDS_DONE_MS", 50.0)
    composer = FakeComposer()
    s, sent = await by_sign_session(tmp_path, [[("COFFEE", 0.99)], [("TEA", 0.3), ("MILK", 0.2), ("WATER", 0.1)]],
                                    composer=composer)
    await push_frames(s, 2)
    await push_frames(s, 2)  # an unsure second sign: the signer drops it
    await s.on_client(PickMsg(segment=1, gloss=""))
    await asyncio.sleep(0.2)
    await s.idle()
    assert composer.calls and composer.calls[0][0] == ["COFFEE"]


async def test_done_pressed_mid_sign_says_the_words_once_it_lands(tmp_path, monkeypatch):
    monkeypatch.setattr(session_mod, "WORDS_DONE_MS", 0.0)
    composer = FakeComposer()
    s, sent = await by_sign_session(tmp_path, [[("COFFEE", 0.99)], [("WANT", 0.97)]], composer=composer)
    await push_frames(s, 2)
    await push_frames(s, 1)  # the next sign has started
    await s.on_client(WordsMsg(action="done"))
    assert not composer.calls
    await push_frames(s, 1)
    await s.idle()
    await s.idle()
    assert composer.calls and composer.calls[0][0] == ["COFFEE", "WANT"]


async def test_done_mid_sign_then_the_sign_abandoned_still_says_the_words(tmp_path, monkeypatch):
    monkeypatch.setattr(session_mod, "WORDS_DONE_MS", 0.0)  # no timer: only Done can say them
    monkeypatch.setattr(session_mod, "FRAME_TIMEOUT_S", 0.05)
    composer = FakeComposer()
    s, sent = await by_sign_session(tmp_path, [[("COFFEE", 0.99)]], composer=composer)
    await push_frames(s, 2)
    await push_frames(s, 1)  # the next sign starts, then the signer walks away
    await s.on_client(WordsMsg(action="done"))
    await asyncio.sleep(0.3)  # the watchdog drops the half-finished turn
    await s.idle()
    assert composer.calls and composer.calls[0][0] == ["COFFEE"] and s.state == "ready"


async def test_by_sign_switched_off_mid_pick_adds_no_word(tmp_path, monkeypatch):
    monkeypatch.setattr(session_mod, "WORDS_DONE_MS", 20.0)
    composer = FakeComposer()
    s, sent = await by_sign_session(tmp_path, [[("TEA", 0.3), ("MILK", 0.2), ("WATER", 0.1)]], turns=1,
                                    composer=composer)
    await push_frames(s, 2)
    await s.on_client(ModeMsg(mode="signing", by_sign=False))
    await s.on_client(PickMsg(segment=0, gloss="MILK"))
    await s.idle()
    await asyncio.sleep(0.1)
    assert not s._words and not composer.calls and s._done_timer is None and s.state == "ready"


async def test_undo_while_the_words_are_said_does_not_cut_them(tmp_path, monkeypatch):
    monkeypatch.setattr(session_mod, "WORDS_DONE_MS", 0.0)
    gate = asyncio.Event()

    class SlowComposer(FakeComposer):
        async def compose(self, *a):
            await gate.wait()
            return await super().compose(*a)

    s, sent = await by_sign_session(tmp_path, [[("COFFEE", 0.99)]], turns=1, composer=SlowComposer())
    await push_frames(s, 2)
    await s.on_client(WordsMsg(action="done"))
    await asyncio.sleep(0.01)
    assert s.state == "reading"
    await s.on_client(WordsMsg(action="undo"))
    await s.on_client(WordsMsg(action="done"))  # already answered: must not linger for the next word
    gate.set()
    await s.idle()
    assert [m for m in sent if m.type == "you_said"] and not s._done_pending and s.state == "ready"


async def test_timer_cannot_start_a_sentence_under_a_new_sign(tmp_path, monkeypatch):
    monkeypatch.setattr(session_mod, "WORDS_DONE_MS", 0.0)
    s, sent = await by_sign_session(tmp_path, [[("COFFEE", 0.99)]], turns=1)
    await push_frames(s, 2)
    await s.on_client(WordsMsg(action="done"))
    assert s.state == "reading"  # set at once, so a TurnStart in the same tick is ignored, not started under it
    await s.idle()


async def test_practice_clears_the_words_and_ignores_done(tmp_path, monkeypatch):
    from server.protocol import PracticeMsg
    monkeypatch.setattr(session_mod, "WORDS_DONE_MS", 20.0)
    composer = FakeComposer()
    s, sent = await by_sign_session(tmp_path, [[("COFFEE", 0.99)]], turns=1, composer=composer)
    await push_frames(s, 2)
    await s.on_client(PracticeMsg(on=True, signs=[]))
    await s.on_client(WordsMsg(action="done"))
    await asyncio.sleep(0.1)
    assert not s._words and not composer.calls


async def test_close_disarms_the_timer(tmp_path, monkeypatch):
    monkeypatch.setattr(session_mod, "WORDS_DONE_MS", 20.0)
    composer = FakeComposer()
    s, sent = await by_sign_session(tmp_path, [[("COFFEE", 0.99)]], turns=1, composer=composer)
    await push_frames(s, 2)
    await s.close()
    await asyncio.sleep(0.1)
    assert not composer.calls


def timed_turn(t0):
    return [TurnStart(), TurnEnd(segments=[seg()], frames=[make_frame(t=t0 + i * 33) for i in range(10)], rest_ms=450)]


async def repeat_session(tmp_path, gap_ms):
    end = 10_000 + 9 * 33
    s, sent = make_session(tmp_path, [*timed_turn(10_000), *timed_turn(end + gap_ms)],
                           [[("SEE", 0.99)], [("SEE", 0.99)]])
    await calibrate(s)
    await s.on_client(ModeMsg(mode="signing", by_sign=True))
    await push_frames(s, 2)
    await push_frames(s, 2)
    return s, sent


async def test_a_quick_repeat_is_one_sign_split_in_two(tmp_path, monkeypatch):
    monkeypatch.setattr(session_mod, "WORDS_DONE_MS", 0.0)
    s, sent = await repeat_session(tmp_path, 700)
    assert [w.gloss for w, _ in s._words] == ["SEE"] and s.state == "ready"
    import json
    rows = [json.loads(line) for line in s.log_path.read_text().splitlines()]
    assert rows[-1]["result"].get("repeat_dropped") is True


async def test_a_repeat_after_a_clear_pause_is_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(session_mod, "WORDS_DONE_MS", 0.0)
    s, sent = await repeat_session(tmp_path, 2000)
    assert [w.gloss for w, _ in s._words] == ["SEE", "SEE"]
