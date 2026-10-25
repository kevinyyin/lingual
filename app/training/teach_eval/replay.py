"""One-sign videos through the app as it runs, scored three ways (rebuilds the harness behind sign_reliability.md).

- isolated: the recognizer's top-5 on the whole clip and on the hands-visible span (eval_recognition.hands_span)
- translator: the browser-rate frames through the real Segmenter (eval_live.run_turn: hands lowered after the clip),
  Session._read_turn (recut and all), then the picker rule of Session._sign_turn (Session._needs_pick) with a scripted signer who picks the
  right sign when it is offered and otherwise presses "None of these"
- lesson: Session._read_practice over the course's signs on the whole attempt and its last 1.5 s (as the lesson page);
  a pass is the target in the top 3 of either read (lessonView PERFORM_TOP_K)
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

from server.protocol import Frame
from server.session import PRACTICE_LATEST_MS, Segment, Session
from server.vocab import load_vocab
from training.eval_live import run_turn
from training.eval_recognition import hands_span

ROOT = Path(__file__).resolve().parents[2]
CURRICULUM = ROOT / "web" / "src" / "teach" / "curriculum.ts"
LESSON_TOP_K = 3


def course_signs(path: Path = CURRICULUM) -> set[str]:
    """Every sign the teaching course uses (the lesson page sends them all as the practice set)."""
    text = path.read_text(encoding="utf-8")
    body = text[text.index("export const COURSE"):]
    return set(re.findall(r'"([A-Z][A-Z0-9-]*)"', body))


@dataclass
class _NullParts:
    """Stand-ins for the parts of a Session a sign read never reaches (voice, Claude, grammar)."""

    calls: list = field(default_factory=list)

    async def send(self, m):
        self.calls.append(m)


def make_session(recognizer, practice: set[str] | None = None, spell_reader=None) -> Session:
    nul = _NullParts()

    class _Readers:
        def read(self, frames, baseline):
            return {"question": None, "negation": False}, {"label": "neutral", "intensity": 0.0}

    s = Session(send=nul.send, vocab=load_vocab(), make_segmenter=lambda b: None, recognizer=recognizer,
                readers=_Readers(), composer=None, voice=None, gloss=None, log_path=Path("nul"), spell_reader=spell_reader)
    s.practice_signs = set(practice or course_signs())
    return s


def isolated(frames: list[Frame], recognizer, k: int = 5) -> dict:
    span = hands_span(frames)
    return {"whole": recognizer.topk(frames, k) if frames else [], "span": recognizer.topk(span, k) if span else []}


def translator(frames: list[Frame], gloss: str, session: Session) -> dict:
    """The turn as a signer gets it. outcome: right_direct | right_picked | wrong_spoken | none_of_these | no_turn |
    no_sign | split_right | split_wrong (a one-sign clip read as several signs; *_right if the sign is among them)."""
    ev = run_turn(frames)
    if ev is None or not ev.segments:
        return {"outcome": "no_turn", "segments": 0, "reads": []}
    read = [(s, t) for s, t in session._read_turn(ev) if t]
    if not read:
        return {"outcome": "no_sign", "segments": 0, "reads": []}
    choices, picked = [], 0
    for s, t in read:
        ask, t = session._needs_pick(t, s)
        if ask:
            offered = [g for g, _ in t[:3]]
            if gloss in offered:  # the scripted signer picks the sign they made
                choices.append(gloss)
                picked += 1
            else:  # "None of these": the turn is dropped
                return {"outcome": "none_of_these", "segments": len(read), "reads": _reads(read)}
        else:
            choices.append(t[0][0])
    if len(read) > 1:
        outcome = "split_right" if gloss in choices else "split_wrong"
    elif choices[0] == gloss:
        outcome = "right_picked" if picked else "right_direct"
    else:
        outcome = "wrong_spoken"
    return {"outcome": outcome, "segments": len(read), "choices": choices, "reads": _reads(read)}


def lesson(frames: list[Frame], gloss: str, session: Session) -> dict:
    """The lesson page's grading of one attempt: whole attempt and its last 1.5 s, course signs only, top 3."""
    ev = run_turn(frames)
    if ev is None or not ev.segments:
        return {"pass": False, "outcome": "no_turn", "reads": []}
    whole = ev.whole or ev.frames
    latest = [f for f in whole if f.t >= whole[-1].t - PRACTICE_LATEST_MS]
    reads = [whole, latest] if len(latest) < len(whole) else [whole]
    tops = [session._read_practice(Segment(frames=fs, kind="sign")) for fs in reads]
    ok = any(gloss in [g for g, _ in t[:LESSON_TOP_K]] for t in tops)
    return {"pass": ok, "outcome": "pass" if ok else "fail", "reads": [[[g, round(p, 4)] for g, p in t[:5]] for t in tops]}


def _reads(read) -> list:
    return [{"kind": s.kind, "t0": s.frames[0].t, "t1": s.frames[-1].t, "top": [[g, round(p, 4)] for g, p in t[:5]]}
            for s, t in read]
