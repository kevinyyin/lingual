from itertools import combinations

from frames import make_frame
from server.demo_sentences import DEFAULT_PATH, DemoSentence, load, snap
from server.protocol import ModeMsg
from server.session import Segment, TurnEnd, TurnStart

from test_session import FakeComposer, calibrate, make_session, push_frames

DEMO = [DemoSentence(("HELLO", "NICE", "MEET"), "Hello, nice to meet you!"),
        DemoSentence(("WANT", "COFFEE", "PLEASE"), "I'd like a coffee, please."),
        DemoSentence(("NEED", "BATHROOM", "WHERE"), "Where is the bathroom? I need it."),
        DemoSentence(("FAVORITE", "JUICE"), "Juice is my favorite.")]


def piece(*guesses):
    """A piece's ranked guesses: the first at 0.7, the rest sharing what is left."""
    return [(g, 0.7 if i == 0 else 0.3 / (len(guesses) - 1)) for i, g in enumerate(guesses)]


def test_pieces_reading_the_sentence_snap_to_it():
    best, score, runner = snap([piece("WANT", "TEA"), piece("COFFEE", "MAKE"), piece("PLEASE", "MY")], DEMO)
    assert best == DEMO[1] and score > runner


def test_two_signs_read_as_one_piece_still_snap():
    best, _, _ = snap([piece("HELLO", "FATHER"), piece("MEET", "NICE")], DEMO)  # NICE and MEET merged
    assert best == DEMO[0]


def test_one_sign_cut_in_two_still_snaps():
    best, _, _ = snap([piece("NEED", "MUST"), piece("BATHROOM", "TOILET"), piece("BATHROOM", "T"), piece("WHERE", "WHAT")], DEMO)
    assert best == DEMO[2]


def test_the_right_sign_a_few_guesses_down_snaps():
    best, _, _ = snap([piece("CHEESE", "BABY", "MILK", "JUICE"), piece("FAVORITE", "TASTE")], DEMO)
    assert best == DEMO[3]


def test_garbage_snaps_to_nothing_so_the_turn_is_translated_as_usual():
    best, _, _ = snap([piece("BABY", "CHEESE"), piece("FATHER", "MOTHER"), piece("SICK", "TIRED")], DEMO)
    assert best is None


def test_no_file_no_sentences_so_the_feature_is_off(tmp_path):
    assert load(tmp_path / "none.yaml") == []
    assert snap([piece("HELLO")], [])[0] is None


def test_the_demo_list_follows_its_rules():
    sentences = load(DEFAULT_PATH)
    assert 10 <= len(sentences) <= 20
    import server.main as M
    for s in sentences:
        assert 2 <= len(s.signs) <= 4 and s.english
        assert set(s.signs) <= set(M.RECOGNIZER.labels), s.signs  # signs the sign model knows
    for a, b in combinations(sentences, 2):  # every two differ in at least 2 signs
        assert len(set(a.signs) ^ set(b.signs)) >= 2, (a.signs, b.signs)


class ByPiece:
    """The translator's reads: piece i's guesses from a script (recognize for the turn, topk for the re-read)."""

    def __init__(self, script):
        self.script, self.i = script, 0

    def recognize(self, seg):
        return self._next()

    def topk(self, frames, k=3):
        return self._next()[:k]

    def _next(self):
        top = self.script[self.i % len(self.script)]
        self.i += 1
        return top


async def run_turn(tmp_path, demo):
    segs = [Segment(frames=[make_frame(t=1000 + i * 20) for i in range(4)], kind="sign"),
            Segment(frames=[make_frame(t=1500 + i * 20) for i in range(4)], kind="sign")]
    events = [TurnStart(), TurnEnd(segments=segs, frames=segs[0].frames + segs[1].frames, rest_ms=0)]
    composer = FakeComposer()
    rec = ByPiece([piece("FAVORITE", "TASTE"), piece("JUICE", "APPLE")] * 2)
    s, sent = make_session(tmp_path, events, recognizer=rec, composer=composer)
    s.demo_sentences = DEMO
    await s.on_client(ModeMsg(mode="signing", demo=demo))
    await calibrate(s)
    await push_frames(s, 2)
    return sent, composer


async def test_demo_mode_says_the_snapped_sentence_without_claude_or_the_picker(tmp_path):
    sent, composer = await run_turn(tmp_path, demo=True)
    (said,) = [m for m in sent if m.type == "you_said"]
    assert said.text == "Juice is my favorite." and [t.gloss for t in said.read] == ["FAVORITE", "JUICE"]
    assert not composer.calls and not [m for m in sent if m.type == "pick_request"]
    assert "speaking" in [m.state for m in sent if m.type == "state"]


async def test_without_demo_mode_the_turn_is_translated_as_before(tmp_path):
    sent, composer = await run_turn(tmp_path, demo=False)
    # as before: unsure reads (0.7 < 0.95) go to the picker, and nothing is said until the signer picks
    (req,) = [m for m in sent if m.type == "pick_request"]
    assert [o.gloss for o in req.options] == ["FAVORITE", "TASTE"]
    assert not [m for m in sent if m.type == "you_said"] and not composer.calls
