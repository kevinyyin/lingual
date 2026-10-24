"""Orchestrator: calibration, turn-taking between the two pipelines, and the turn log."""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Literal, Protocol, TypedDict

import numpy as np

from .protocol import (
    AvatarDoneMsg, CalibrateMsg, CaptionMsg, FrameMsg, ModeMsg, PickMsg, PickOption, PickRequestMsg, PracticeMsg, SayMsg,
    RecognizedMsg, SignPlanMsg, SignToken, StateMsg, YouSaidMsg, Frame,
)
from .vocab import Vocab

# A sign read less surely than this goes to the picker instead of being spoken. One-sign replays of all 160 signs
# (WLASL videos): at 0.6 a wrong word was spoken in 33% of turns, at 0.95 in 15%, the picker showing in 43% (was 16%).
PICK_THRESHOLD = 0.95
SPELL_PICK_THRESHOLD = 0.6  # the letter reader's score (mean letter confidence) has its own scale: not re-tuned
SAY_EMOTION = {"label": "happy", "intensity": 0.4}  # the plushie's lesson lines: warm, not excited
PRACTICE_TOP_K = 5  # guesses per turn sent to teaching mode, which grades on its own
CALIBRATION_MS = 2000.0  # by time: the browser sends 8-15 frames/s, so 60 frames took 4-8 s
REST_MARGIN = 0.05  # rest line sits this far above resting wrists (image y grows downward)
FRAME_TIMEOUT_S = 3.0  # no frames this long mid-turn (signer left the frame) ends the turn
# A lesson attempt is also read as its last 1.5 s alone, so a wrong sign corrected without resting counts the correction.
# Replayed WLASL course signs, a wrong sign + 300 ms pause + the right one: the right one passes 79% (whole turn alone
# 44%); a correct single sign 86% (81%); a wrong one 3.6% (1.5%). Cutting at the segmenter's pauses instead did as well
# only after a long (600 ms+) pause.
PRACTICE_LATEST_MS = 1500.0


class Grammar(TypedDict):
    question: Literal["yn", "wh"] | None
    negation: bool


class Emotion(TypedDict):
    label: Literal["happy", "sad", "angry", "neutral"]
    intensity: float


@dataclass
class Baseline:
    rest_y: float | tuple[float, float]  # wrist y above which (smaller y) a hand counts as raised; per wrist (15, 16)
    brow: dict[str, float]  # resting brow* blendshapes
    motion_floor: float  # mean wrist speed at rest, per frame
    speed: float | None = None  # mean signing wrist speed, set from the first turn (emotion.py)
    size: float | None = None  # mean signing wrist box area, set from the first turn (emotion.py)


@dataclass
class Segment:
    frames: list[Frame]
    kind: Literal["sign", "spell"]
    # the sign plus the quieter frames around it (segmenter.CONTEXT_MS): published sign models were trained on whole
    # dictionary clips with the hands rising and settling, and score far better given them
    context: list[Frame] | None = None


@dataclass
class TurnStart:
    pass


@dataclass
class TurnEnd:
    segments: list[Segment]
    frames: list[Frame]  # the whole turn, for the face readers
    rest_ms: float  # how long ago (browser clock) the hands came to rest
    whole: list[Frame] = field(default_factory=list)  # first sign's context to the last's: the turn read as one sign


SegEvent = TurnStart | TurnEnd


class Segmenter(Protocol):
    def push(self, f: Frame) -> SegEvent | None: ...


class Recognizer(Protocol):
    def recognize(self, seg: Segment) -> list[tuple[str, float]]: ...

    def topk(self, frames: list[Frame], k: int) -> list[tuple[str, float]]: ...


class Readers(Protocol):
    def read(self, frames: list[Frame], baseline: Baseline) -> tuple[Grammar, Emotion]: ...


class Composer(Protocol):
    async def compose(self, signs: list[str], spelled: list[str], grammar: Grammar, emotion: Emotion) -> str: ...


class Voice(Protocol):
    def speak(self, text: str, emotion: Emotion) -> AsyncIterator[str]: ...


class Gloss(Protocol):
    async def plan(self, english: str) -> SignPlanMsg: ...


def _wrists(f: Frame) -> np.ndarray:
    return np.asarray([f.pose[15][:2], f.pose[16][:2]], dtype=np.float32)


def compute_baseline(frames: list[Frame]) -> Baseline:
    w = np.stack([_wrists(f) for f in frames])
    speed = np.linalg.norm(np.diff(w, axis=0), axis=-1).mean() if len(w) > 1 else 0.0
    faces = [f.blend for f in frames if f.blend]
    keys = {k for b in faces for k in b if k.startswith("brow")}
    brow = {k: float(np.mean([b.get(k, 0.0) for b in faces])) for k in keys}
    # each wrist's own resting height (hands may rest at different heights); the median ignores a hand moved briefly
    left, right = np.median(w[:, :, 1], axis=0) - REST_MARGIN
    return Baseline(rest_y=(float(left), float(right)), brow=brow, motion_floor=float(speed))


def now_ms() -> float:
    return time.monotonic() * 1000


@dataclass
class _Turn:
    direction: str
    stamps: dict[str, float] = field(default_factory=dict)
    result: Any = None
    error: str | None = None


class Session:
    def __init__(self, send: Callable[[Any], Awaitable[None]], vocab: Vocab,
                 make_segmenter: Callable[[Baseline], Segmenter], recognizer: Recognizer, readers: Readers,
                 composer: Composer, voice: Voice, gloss: Gloss, log_path: Path | str, spell_reader: Recognizer | None = None):
        self.send, self.vocab = send, vocab
        self.make_segmenter, self.recognizer, self.readers = make_segmenter, recognizer, readers
        self.composer, self.voice, self.gloss = composer, voice, gloss
        self.spell_reader = spell_reader
        self.log_path = Path(log_path)
        self.state = "calibrating"
        self.baseline: Baseline | None = None
        self.segmenter: Segmenter | None = None
        self._calib: list[Frame] = []
        self._task: asyncio.Task | None = None
        self._pick: asyncio.Future | None = None
        self._offered: tuple[int, set[str]] | None = None  # (segment, glosses) the picker shows; "" = none of these
        self._last_frame = 0.0
        self._pending_speech: _Turn | None = None
        self._turn_ids = 0
        self.practice = False
        self.practice_signs: set[str] = set()
        self.mode: str | None = None  # the translator's switch (ModeMsg): "voice" or "signing"; None runs both
        self._say_lock = asyncio.Lock()
        self._says: set[asyncio.Task] = set()  # asyncio keeps only weak refs to tasks
        self._say_waiting: asyncio.Task | None = None  # the one line queued behind the playing one

    # --- plumbing ---

    async def _set(self, state: str) -> None:
        self.state = state
        await self.send(StateMsg(state=state))

    def _spawn(self, coro) -> None:
        self._task = asyncio.create_task(coro)

    async def _idle(self) -> None:
        """End of a speech turn: ready, or back to calibrating if the camera hasn't calibrated yet."""
        if self.baseline is None:
            self._calib = []
            await self._set("calibrating")
        else:
            await self._set("ready")

    def _busy(self) -> bool:
        """A turn is being processed: new turns in either direction are dropped until it finishes."""
        return self._task is not None and not self._task.done()

    async def _watchdog(self) -> None:
        while self.state == "signing" and not self._busy():
            await asyncio.sleep(min(FRAME_TIMEOUT_S / 2, 0.5))
            if self.state == "signing" and not self._busy() and time.monotonic() - self._last_frame > FRAME_TIMEOUT_S:
                self.segmenter = self.make_segmenter(self.baseline)  # drop the half-finished turn
                await self._set("ready")

    async def idle(self) -> None:
        """Wait for the turn in flight (tests and shutdown)."""
        while self._task and not self._task.done():
            await asyncio.sleep(0)
            if self._pick and not self._pick.done():
                return  # waiting on the signer's pick
        if self._task:
            await self._task

    def _log(self, turn: _Turn) -> None:
        self._turn_ids += 1
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as fh:
            fh.write(json.dumps({"turn_id": self._turn_ids, "direction": turn.direction, "stamps": turn.stamps,
                                 "result": turn.result, "error": turn.error}) + "\n")

    # --- inputs ---

    async def close(self) -> None:
        """The page went away: its turn and the lines it queued must not keep using the voice (or the dead socket)."""
        tasks = [*self._says, *([self._task] if self._busy() else [])]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _say(self, text: str) -> None:
        """Lines never overlap and never change the session state; a failed line is only logged."""
        async with self._say_lock:
            if self._say_waiting is asyncio.current_task():
                self._say_waiting = None
            try:
                async for _ in self.voice.speak(text, SAY_EMOTION):
                    pass
            except Exception as e:
                logging.getLogger("asl.session").warning("say failed: %s", e)

    async def on_client(self, msg) -> None:
        if isinstance(msg, CalibrateMsg):
            if self._busy():  # recalibrating abandons the turn in flight (e.g. a picker nobody can answer)
                self._task.cancel()
                await asyncio.gather(self._task, return_exceptions=True)
            self._calib = []
            await self._set("calibrating")
        elif isinstance(msg, FrameMsg):
            await self._on_frame(msg)
        elif isinstance(msg, ModeMsg):
            self.mode = msg.mode
            if msg.mode == "voice" and self.state in ("signing", "picking"):
                await self._yield_to_speech()  # a sign turn in flight is dropped
            elif msg.mode == "signing" and self.state == "listening":
                await self._idle()
        elif isinstance(msg, PracticeMsg):
            self.practice, self.practice_signs = msg.on, set(msg.signs)
        elif isinstance(msg, SayMsg) and self.practice:  # translator speech is the session's own; never page-chosen
            if self._say_waiting:  # a newer line makes the waiting one stale: never more than one in line
                self._say_waiting.cancel()
            t = self._say_waiting = asyncio.create_task(self._say(msg.text))
            self._says.add(t)
            t.add_done_callback(self._says.discard)
        elif isinstance(msg, PickMsg):
            offered = self._offered and msg.segment == self._offered[0] and msg.gloss in self._offered[1]
            if offered and self._pick and not self._pick.done():
                self._pick.set_result((msg.segment, msg.gloss))
        elif isinstance(msg, AvatarDoneMsg) and self.state == "avatar":
            turn, self._pending_speech = self._pending_speech, None
            if turn:
                self._log(turn)
            await self._idle()

    async def _on_frame(self, f: Frame) -> None:
        if self.mode == "voice":
            return  # the camera is off in voice mode (the page stops sending too)
        if self.state == "calibrating":
            self._calib.append(f)
            if self._calib[-1].t - self._calib[0].t >= CALIBRATION_MS:
                self.baseline = compute_baseline(self._calib)
                self.segmenter = self.make_segmenter(self.baseline)
                await self._set("ready")
            return
        if self.state not in ("ready", "signing") or self.segmenter is None or self._busy():
            return  # one direction at a time: frames while a turn is processed or output plays are ignored
        self._last_frame = time.monotonic()
        ev = self.segmenter.push(f)
        if isinstance(ev, TurnStart) and self.state == "ready":
            await self._set("signing")
            self._watch = asyncio.create_task(self._watchdog())  # kept referenced so it is not garbage-collected
        elif isinstance(ev, TurnEnd) and self.state == "signing":
            self._spawn(self._sign_turn(ev))

    async def _yield_to_speech(self) -> None:
        """Finished words win over a sign turn still being read or picked: on one laptop that turn is often the hearing
        person's own hands moving as they talk, and dropping their sentence left the avatar still with no feedback."""
        if self._busy():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        if self.baseline is not None:
            self.segmenter = self.make_segmenter(self.baseline)
        await self._set("ready")

    async def on_transcript(self, text: str, final: bool) -> None:
        if self.practice or self.mode == "signing":
            return  # lessons and signing mode are sign -> page only; the mic would start speech turns mid-sign
        if final and text.strip() and self.state in ("signing", "picking"):
            await self._yield_to_speech()  # never while voice or avatar output plays (those states drop speech)
        # Calibration is only for the signer's camera; speech -> sign doesn't need it.
        if self.state not in ("ready", "listening", "calibrating") or self._busy():
            return  # signer turn, voice, avatar or a turn being planned: drop the audio
        if not text.strip():
            if final and self.state == "listening":
                await self._idle()
            return
        if self.state in ("ready", "calibrating"):
            await self._set("listening")
        if not final:
            await self.send(CaptionMsg(text=text, final=False))
            return
        self._spawn(self._speech_turn(text))

    # --- turns ---

    def _dump(self, ev: TurnEnd) -> None:
        """ASL_DUMP_TURNS=1: every sign turn's landmarks (pre-roll included) to logs/turn_frames/, for diagnosing
        recognition on a real webcam (training.eval_live reads the same Frame fields)."""
        frames = sorted({f.t: f for f in [*ev.frames, *(f for s in ev.segments for f in (s.context or []))]}.values(),
                        key=lambda f: f.t)
        out = self.log_path.parent / "turn_frames" / f"{int(time.time() * 1000)}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"practice": self.practice, "frames": [f.model_dump() for f in frames],
                                   "segments": [[s.frames[0].t, s.frames[-1].t, s.kind] for s in ev.segments]}))

    async def _sign_turn(self, ev: TurnEnd) -> None:
        turn = _Turn("sign2speech")
        turn.stamps["hands_rest"] = now_ms() - ev.rest_ms
        if os.environ.get("ASL_DUMP_TURNS"):
            self._dump(ev)
        try:
            if not ev.segments:
                return
            if self.practice:  # one sign per exercise: the hands going down, or a slow sign cut in two, is not a second sign
                whole = ev.whole or ev.frames  # read whole, and its last PRACTICE_LATEST_MS alone; the page takes either
                latest = [f for f in whole if f.t >= whole[-1].t - PRACTICE_LATEST_MS]
                reads = [Segment(frames=fs, kind="sign") for fs in ([whole, latest] if len(latest) < len(whole) else [whole])]
                tops = [await asyncio.to_thread(self._read_practice, r) for r in reads]
                turn.stamps["recognized"] = now_ms()
                turn.result = {"top3": [[[g, round(p, 3)] for g, p in t[:3]] for t in tops]}
                await self.send(RecognizedMsg(segments=[
                    [PickOption(gloss=g, english=self._english(g, r), p=p) for g, p in t] for r, t in zip(reads, tops)]))
                return
            read = [(s, await asyncio.to_thread(self._read_segment, s)) for s in ev.segments]
            read = [(s, t) for s, t in read if t]  # unreadable segments (e.g. no confident letters) are dropped
            segments, tops = [s for s, _ in read], [t for _, t in read]
            grammar, emotion = await asyncio.to_thread(self.readers.read, ev.frames, self.baseline)
            turn.stamps["recognized"] = now_ms()
            turn.result = {"top3": [[[g, round(p, 3)] for g, p in t[:3]] for t in tops]}  # logged even if cancelled
            choice = [t[0][0] for t in tops]
            picks = 0
            for i, t in enumerate(tops):
                spelled = segments[i].kind == "spell" and self.spell_reader is not None  # read by the letter reader
                if t[0][1] < (SPELL_PICK_THRESHOLD if spelled else PICK_THRESHOLD):
                    await self._set("picking")
                    self._pick = asyncio.get_running_loop().create_future()
                    await self.send(PickRequestMsg(segment=i, options=[
                        PickOption(gloss=g, english=self._english(g, segments[i]), p=p) for g, p in t[:3]]))
                    self._offered = (i, {g for g, _ in t[:3]} | {""})
                    _, choice[i] = await self._pick
                    if not choice[i]:
                        turn.error = "cancelled by signer"
                        return
                    picks += 1
            turn.stamps["picked"] = now_ms()
            if not segments:
                return
            signs = [c for c, s in zip(choice, segments) if s.kind == "sign"]
            spelled = [c for c, s in zip(choice, segments) if s.kind == "spell"]
            text = await self.composer.compose(signs, spelled, grammar, emotion)
            turn.stamps["composed"] = now_ms()
            turn.result = {"signs": signs, "spelled": spelled, "grammar": grammar, "emotion": emotion, "text": text, "picks": picks,
                           "top3": turn.result["top3"]}
            if not text:  # the signs make no sentence: say so on screen, speak nothing
                await self.send(YouSaidMsg(text="", emotion=emotion["label"], unclear=True))
                return
            await self.send(YouSaidMsg(text=text, emotion=emotion["label"]))
            await self._set("speaking")
            async for e in self.voice.speak(text, emotion):
                if e == "first_audio":
                    turn.stamps["first_audio"] = now_ms()
        except asyncio.CancelledError:
            turn.error = "cancelled"
        except Exception as e:  # never leave the session stuck mid-turn
            turn.error = f"{type(e).__name__}: {e}"
        finally:
            self._pick, self._offered = None, None
            if ev.segments:
                self._log(turn)
            await self._set("ready")

    def _read_segment(self, s: Segment) -> list[tuple[str, float]]:
        if s.kind == "spell" and self.spell_reader is not None:
            return self.spell_reader.recognize(s)
        return self.recognizer.recognize(s)

    def _read_practice(self, s: Segment) -> list[tuple[str, float]]:
        """Guesses among the course's signs only: the lesson knows the sign it asked for, and among all ~2,900 signs
        rare look-alikes beat it (correct signing passed 48% of the time, 88% among the course's)."""
        top = self.recognizer.topk(s.frames, 10**6)
        if self.practice_signs:
            top = [(g, p) for g, p in top if g in self.practice_signs]
        z = sum(p for _, p in top) or 1.0
        return [(g, p / z) for g, p in top[:PRACTICE_TOP_K]]

    def _english(self, gloss: str, s: Segment) -> str:
        sign = self.vocab.get(gloss)
        return sign.english if sign and s.kind == "sign" else gloss

    async def _speech_turn(self, text: str) -> None:
        turn = _Turn("speech2sign")
        turn.stamps["speech_end"] = now_ms()
        await self.send(CaptionMsg(text=text, final=True))
        try:
            plan = await self.gloss.plan(text)
            turn.stamps["planned"] = now_ms()
            turn.result = plan.model_dump()
            await self.send(plan)
            turn.stamps["avatar_start"] = now_ms()
            self._pending_speech = turn
            await self._set("avatar")
        except Exception as e:
            turn.error = f"{type(e).__name__}: {e}"
            self._log(turn)
            await self._idle()


# --- stubs: replaced by real parts in later tasks ---

class StubSegmenter:
    def __init__(self, baseline: Baseline, rest_frames: int = 20):
        self.b, self.rest_frames = baseline, rest_frames
        self.turn: list[Frame] | None = None
        self.still = 0

    def push(self, f: Frame) -> SegEvent | None:
        raised = bool((_wrists(f)[:, 1] < self.b.rest_y).any())
        if self.turn is None:
            if raised:
                self.turn, self.still = [f], 0
                return TurnStart()
            return None
        self.turn.append(f)
        self.still = 0 if raised else self.still + 1
        if self.still < self.rest_frames:
            return None
        frames, self.turn = self.turn, None
        rest_ms = frames[-1].t - frames[-self.rest_frames].t
        return TurnEnd(segments=[Segment(frames=frames, kind="sign")], frames=frames, rest_ms=rest_ms)


class StubRecognizer:
    def recognize(self, seg: Segment) -> list[tuple[str, float]]:
        return [("HELLO", 0.9), ("THANK-YOU", 0.05), ("PLEASE", 0.05)]


# A short café conversation in ASL order; the demo recognizer reads one gloss per signed segment.
DEMO_SCRIPT = [
    ["HELLO", "NICE", "MEET", "YOU"],
    ["YOU", "NAME", "WHAT"],
    ["I", "WANT", "COFFEE", "MILK"],
    ["YOU", "HAVE", "TEA"],
    ["I", "NOT", "HUNGRY"],
    ["COFFEE", "EXPENSIVE", "WHY"],
    ["THANK-YOU", "GOODBYE"],
]
DEMO_TURN_GAP_MS = 1000  # a gap this long between segments starts the script's next sentence


class DemoRecognizer:
    """Stand-in until models/signs.pt is trained: the n-th segment of a turn reads as the n-th gloss of the next
    scripted sentence, so compose, grammar, emotion and voice all get varied, realistic input."""

    def __init__(self, script: list[list[str]] = DEMO_SCRIPT):
        self.script, self.sentence, self.word, self.last_t = script, -1, 0, None

    def recognize(self, seg: Segment) -> list[tuple[str, float]]:
        if self.last_t is None or seg.frames[0].t - self.last_t > DEMO_TURN_GAP_MS:
            self.sentence, self.word = (self.sentence + 1) % len(self.script), 0
        self.last_t = seg.frames[-1].t
        words = self.script[self.sentence]
        gloss = words[self.word % len(words)]
        self.word += 1
        alts = [g for g in ("PLEASE", "THANK-YOU", "HELLO") if g != gloss][:2]
        return [(gloss, 0.9), *((g, 0.05) for g in alts)]

    def topk(self, frames: list[Frame], k: int = 3) -> list[tuple[str, float]]:
        return self.recognize(Segment(frames=frames, kind="sign"))[:k]


class StubReaders:
    def read(self, frames, baseline) -> tuple[Grammar, Emotion]:
        return {"question": None, "negation": False}, {"label": "neutral", "intensity": 0.0}


class StubComposer:
    async def compose(self, signs, spelled, grammar, emotion) -> str:
        return " ".join(signs + spelled).lower()


class StubVoice:
    async def speak(self, text, emotion):
        print(f"[voice] {text}")
        yield "first_audio"
        yield "done"


class StubGloss:
    async def plan(self, english: str) -> SignPlanMsg:
        words = ["".join(c for c in w.upper() if c.isalpha()) for w in english.split()]
        return SignPlanMsg(tokens=[SignToken(kind="spell", word=w) for w in words if w],
                           grammar="statement", tone="neutral", english=english)
