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

from . import plush
from .protocol import (
    AnswerMsg, AvatarDoneMsg, BuddyMsg, CalibrateMsg, CaptionMsg, FrameMsg, ModeMsg, PickMsg, PickOption, PickRequestMsg, PracticeMsg, RedoMsg,
    SayMsg, RecognizedMsg, SignPlanMsg, SignToken, StateMsg, TalkMsg, WordsMsg, WordsUpdateMsg, YouSaidMsg, Frame,
)
from .vocab import Vocab
from .voice import BUDDY_VOICES

# A sign read less surely than this goes to the picker instead of being spoken; a sign model may set its own
# (pick_threshold: scores differ between models). This one is the ASL Citizen ST-GCN's: one-sign replays of all 160
# signs (WLASL videos): at 0.6 a wrong word was spoken in 33% of turns, at 0.95 in 15%, the picker showing in 43%.
PICK_THRESHOLD = 0.95
# A piece taken for spelling whose letter read is shorter or less sure than this goes to the sign model instead: the
# Core ML reader gave mislabelled WLASL signs no letters or one unsure one (2% read 2+ letters at 0.5+: BUS, JULY,
# often spelled anyway); Kaggle letter-only phrases always read 2+ letters at 0.5+.
SPELL_MIN_LETTERS, SPELL_MIN_P = 2, 0.5
SPELL_PICK_THRESHOLD = 0.6  # the letter reader's score (mean letter confidence) has its own scale: not re-tuned
# Look-alikes: two signs ASL-LEX gives the same handshape and major location (GOOD/THANK-YOU, MINUTE/HOUR), which differ
# only in movement or orientation, the parameters the sign model reads least surely. A top-1 that would be spoken goes
# to the picker when a look-alike of it in the top 3 has p >= LOOKALIKE_RATIO x p1 (and p1 < LOOKALIKE_CAP); a picker
# that offers none has its 3rd option swapped for the best look-alike if p >= LOOKALIKE_SWAP_FLOOR x p3 (and >= 0.01).
# One-sign replays (teach_eval quick_eval, 995 clips): 3 more right, none broken, ~4 right ones now picked, not spoken.
LOOKALIKES = Path(__file__).resolve().parent.parent / "vocab" / "lookalikes.json"  # training.build_lookalikes
LOOKALIKE_RATIO, LOOKALIKE_CAP = 0.25, 0.9
LOOKALIKE_SWAP_FLOOR, LOOKALIKE_MIN_P = 0.5, 0.01
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
# Sign by sign: the words are said this long after the last one is read, unless another sign starts (0: only on Done).
WORDS_DONE_MS = 3000.0
# Sign by sign: the same word again this soon after the last one ended is one sign split in two turns (the replay's
# commonest error: "SEE SEE"); a real repeat needs a clear pause.
REPEAT_GAP_MS = 1200.0


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


def load_lookalikes(path: Path | str | None = None) -> dict[str, tuple[str, str]]:
    """Gloss -> (handshape, major location); empty (no look-alike guard) when the file is missing or unreadable."""
    try:
        return {g: tuple(k) for g, k in json.loads(Path(path or LOOKALIKES).read_text(encoding="utf-8")).items()}
    except (OSError, ValueError) as e:
        logging.getLogger(__name__).warning("no look-alike table (%s): look-alike guard off", e)
        return {}


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
        self.pick_threshold = getattr(recognizer, "pick_threshold", PICK_THRESHOLD)
        labels = set(getattr(recognizer, "labels", None) or [])
        self._alike = {g: k for g, k in load_lookalikes().items() if g in labels} if labels else {}
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
        self.talking = False  # push to talk held (TalkMsg)
        self.by_sign = False  # sign by sign (ModeMsg.by_sign): one word per turn, said together on Done
        self._words: list[tuple[SignToken, list[Frame]]] = []  # the words read so far, with their turns' frames
        self._done_timer: asyncio.Task | None = None
        self._done_pending = False  # Done pressed while a word was being signed or read: say them once it lands
        self._saying = False  # the words are being composed and said (_say_words)
        self._closed = False
        self._say_lock = asyncio.Lock()
        self._says: set[asyncio.Task] = set()  # asyncio keeps only weak refs to tasks

    def hears_speech(self) -> bool:
        """The mic's gate (stt.MicTranscriber sends silence otherwise): never over our own voice or avatar output or in
        signing mode, and only while push to talk is held; words during a camera turn take over (on_transcript)."""
        return self.talking and self.state not in ("speaking", "avatar") and self.mode != "signing"

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
                self._after_word()  # sign by sign: the words before it are still said (now, if Done came mid-turn)

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
        self._closed = True
        self._disarm_done()  # first: a timer firing during the gather below would start a new turn on the dead socket
        tasks = [*self._says, *([self._task] if self._busy() else [])]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _say(self, text: str) -> None:
        """Lines never overlap and never change the session state; a failed line is only logged."""
        async with self._say_lock:  # a cut-off line is still silencing its speaker: the next starts after that
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
            await self._clear_words()
            await self._set("calibrating")
        elif isinstance(msg, FrameMsg):
            await self._on_frame(msg)
        elif isinstance(msg, ModeMsg):
            self.mode = msg.mode
            self.by_sign = msg.by_sign and msg.mode == "signing"
            if not self.by_sign:
                await self._clear_words()
            if msg.mode == "voice" and self.state in ("signing", "reading", "picking"):
                await self._yield_to_speech()  # a sign turn in flight is dropped
            elif msg.mode == "signing" and self.state == "listening":
                await self._idle()
        elif isinstance(msg, TalkMsg):
            self.talking = msg.on
        elif isinstance(msg, BuddyMsg):
            self.voice.voice_id = BUDDY_VOICES[msg.buddy]  # the next line is said in it
        elif isinstance(msg, RedoMsg) and self.practice:
            if self._busy():  # the attempt still being read is dropped unread
                self._task.cancel()
                await asyncio.gather(self._task, return_exceptions=True)
            # mid-turn, or a read cancelled before it began (its finally, which says "ready", never ran): start over
            if self.state in ("signing", "reading") and self.baseline is not None:
                self.segmenter = self.make_segmenter(self.baseline)
                await self._set("ready")
        elif isinstance(msg, WordsMsg) and self.by_sign and not self.practice:
            await self._on_words(msg.action)
        elif isinstance(msg, PracticeMsg):
            self.practice, self.practice_signs = msg.on, set(msg.signs)
            if msg.on:
                await self._clear_words()  # a lesson must not say translator words on a timer
        elif isinstance(msg, SayMsg) and self.practice:  # translator speech is the session's own; never page-chosen
            for t in self._says:  # the lesson moved on: a line still playing or waiting is stale, so the newest cuts it off
                t.cancel()
            t = asyncio.create_task(self._say(msg.text))
            self._says.add(t)
            t.add_done_callback(self._says.discard)
        elif isinstance(msg, AnswerMsg) and self.practice:
            plush.react(msg.result)
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
        if os.environ.get("ASL_DEBUG_FRAMES"):  # TEMP debug: once a second, what the camera stream looks like here
            self._dbg = getattr(self, "_dbg", 0) + 1
            if time.monotonic() - getattr(self, "_dbg_t", 0.0) >= 1.0:
                self._dbg_t = time.monotonic()
                seg = self.segmenter
                raised = seg._raised(f) if seg is not None and hasattr(seg, "_raised") else None
                logging.getLogger("asl.session").warning(
                    "frames=%d state=%s mode=%s by_sign=%s busy=%s hands L/R=%s/%s raised=%s wristY=%.2f/%.2f rest_y=%s",
                    self._dbg, self.state, self.mode, self.by_sign, self._busy(), f.left is not None, f.right is not None,
                    raised, f.pose[15][1], f.pose[16][1], getattr(self.baseline, "rest_y", None))
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
            self._disarm_done()  # another word is coming
            await self._set("signing")
            self._watch = asyncio.create_task(self._watchdog())  # kept referenced so it is not garbage-collected
        elif isinstance(ev, TurnEnd) and self.state == "signing":
            await self._set("reading")  # the page freezes its camera: what comes now is not part of this sign
            self._spawn(self._word_turn(ev) if self.by_sign and not self.practice else self._sign_turn(ev))

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
        if final and text.strip() and self.state in ("signing", "reading", "picking"):
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
            read = await asyncio.to_thread(self._read_turn, ev)
            read = [(s, t) for s, t in read if t]  # unreadable segments (e.g. no confident letters) are dropped
            segments, tops = [s for s, _ in read], [t for _, t in read]
            grammar, emotion = await asyncio.to_thread(self.readers.read, ev.frames, self.baseline)
            turn.stamps["recognized"] = now_ms()
            turn.result = {"top3": [[[g, round(p, 3)] for g, p in t[:3]] for t in tops]}  # logged even if cancelled
            choice = [t[0][0] for t in tops]
            picks, offered = 0, []
            asks = await asyncio.to_thread(lambda: [self._needs_pick(t, s) for s, t in zip(segments, tops)])
            for i, (reason, t) in enumerate(asks):
                if reason:
                    offered.append({"segment": i, "reason": reason, "lookalike": reason != "unsure", "options": [[g, round(p, 3)] for g, p in t[:3]]})
                    turn.result["offered"] = offered
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
                           "top3": turn.result["top3"], "offered": offered}
            read = [SignToken(kind="sign", gloss=c) if s.kind == "sign" else SignToken(kind="spell", word=c)
                    for c, s in zip(choice, segments)]
            if not text:  # the signs make no sentence: say so on screen, speak nothing
                await self.send(YouSaidMsg(text="", emotion=emotion["label"], unclear=True, read=read))
                return
            await self.send(YouSaidMsg(text=text, emotion=emotion["label"], read=read))
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

    # --- sign by sign ---

    def _read_word(self, ev: TurnEnd) -> tuple[Segment, list[tuple[str, float]]]:
        """The turn as one word: a spelled word by the letter reader, otherwise the whole turn read as one sign. No
        cutting: in this mode the hands going down between words is the cut (teaching clips, one sign per turn: whole
        turn 90.9% right with the picker, cut pieces 88.8%)."""
        if len(ev.segments) == 1 and ev.segments[0].kind == "spell":
            s = ev.segments[0]
            return s, self._read_segment(s)
        s = Segment(frames=ev.whole or ev.frames, kind="sign")
        return s, self.recognizer.recognize(s)

    async def _word_turn(self, ev: TurnEnd) -> None:
        """Sign by sign: one word, checked like any sign (the picker when unsure, where "None of these" drops only this
        word), then added to the words on screen."""
        turn = _Turn("sign2speech")
        turn.stamps["hands_rest"] = now_ms() - ev.rest_ms
        try:
            if not ev.segments:
                return
            seg, t = await asyncio.to_thread(self._read_word, ev)
            turn.stamps["recognized"] = now_ms()
            turn.result = {"by_sign": True, "top3": [[g, round(p, 3)] for g, p in t[:3]]}
            if not t:
                return
            reason, t = await asyncio.to_thread(self._needs_pick, t, seg)
            word = t[0][0]
            if reason:
                i = len(self._words)
                await self._set("picking")
                self._pick = asyncio.get_running_loop().create_future()
                await self.send(PickRequestMsg(segment=i, options=[
                    PickOption(gloss=g, english=self._english(g, seg), p=p) for g, p in t[:3]]))
                self._offered = (i, {g for g, _ in t[:3]} | {""})
                _, word = await self._pick
                turn.result["picked"] = word
                if not word:
                    turn.error = "dropped by signer"
                    return
            if not self.by_sign:  # sign by sign was switched off while this word was read
                turn.error = "by_sign off"
                return
            token = SignToken(kind="spell", word=word) if seg.kind == "spell" else SignToken(kind="sign", gloss=word)
            if self._words and self._words[-1][0] == token and self._words[-1][1] and ev.frames                     and ev.frames[0].t - self._words[-1][1][-1].t < REPEAT_GAP_MS:
                turn.result["repeat_dropped"] = True  # one sign split into two turns, not a word said twice
                return
            self._words.append((token, ev.frames))
            turn.result["word"] = word
            await self.send(WordsUpdateMsg(words=[w for w, _ in self._words]))
        except asyncio.CancelledError:
            turn.error = "cancelled"
        except Exception as e:  # never leave the session stuck mid-turn
            turn.error = f"{type(e).__name__}: {e}"
        finally:
            self._pick, self._offered = None, None
            if ev.segments:
                self._log(turn)
            # a word or not (a fidget, "None of these"), the words so far are said after the pause, or now if Done
            # came mid-turn; not when the turn was cancelled (Undo, the page closing: those decide themselves)
            if turn.error != "cancelled" and self._done_pending and self._words and self.by_sign:
                self._done_pending = False
                await self._say_words()  # in this task: no gap where a new sign could start under the sentence
            else:
                await self._set("ready")
                if turn.error != "cancelled":
                    self._arm_done()

    async def _say_words(self) -> None:
        """Sign by sign: the words so far, composed into a sentence and said, as a whole turn's would be."""
        turn = _Turn("sign2speech")
        words, self._words = self._words, []
        self._saying, self._done_pending = True, False
        await self._set("reading")  # composing: no new sign until it is said
        await self.send(WordsUpdateMsg(words=[]))
        try:
            signs = [w.gloss for w, _ in words if w.kind == "sign"]
            spelled = [w.word for w, _ in words if w.kind == "spell"]
            frames = [f for _, fs in words for f in fs]
            grammar, emotion = await asyncio.to_thread(self.readers.read, frames, self.baseline)
            text = await self.composer.compose(signs, spelled, grammar, emotion)
            turn.stamps["composed"] = now_ms()
            turn.result = {"by_sign": True, "signs": signs, "spelled": spelled, "grammar": grammar, "emotion": emotion,
                           "text": text}
            read = [w for w, _ in words]
            if not text:  # the signs make no sentence: say so on screen, speak nothing
                await self.send(YouSaidMsg(text="", emotion=emotion["label"], unclear=True, read=read))
                return
            await self.send(YouSaidMsg(text=text, emotion=emotion["label"], read=read))
            await self._set("speaking")
            async for e in self.voice.speak(text, emotion):
                if e == "first_audio":
                    turn.stamps["first_audio"] = now_ms()
        except asyncio.CancelledError:
            turn.error = "cancelled"
        except Exception as e:  # never leave the session stuck mid-turn
            turn.error = f"{type(e).__name__}: {e}"
        finally:
            self._saying, self._done_pending = False, False  # Done pressed while it was said is already answered
            self._log(turn)
            await self._set("ready")

    def _start_saying(self) -> None:
        self.state = "reading"  # now, not when the task first runs: a TurnStart in between would start a sign under it
        self._spawn(self._say_words())

    def _after_word(self) -> None:
        """A turn ended without a word reading still in flight: say the words now if Done came mid-turn, else after the
        pause."""
        if self._done_pending and self._words and self.state == "ready" and not self._busy():
            self._done_pending = False
            self._start_saying()
        else:
            self._arm_done()

    async def _on_words(self, action: str) -> None:
        if self._saying:
            return  # the words are already being said: Undo/Done/Clear have nothing left to act on
        self._disarm_done()
        self._done_pending = False
        if action == "done":
            if self.state in ("signing", "reading", "picking") or self._busy():  # e.g. the hand reaching for the key
                self._done_pending = True  # started a turn; or a word turn is still finishing
            elif self._words and self.state == "ready":
                self._start_saying()
        elif action == "clear":
            await self._clear_words()
        elif self.state in ("signing", "reading", "picking"):  # undo the word being signed or read: dropped unread
            if self._busy():
                self._task.cancel()
                await asyncio.gather(self._task, return_exceptions=True)
            if self.baseline is not None:
                self.segmenter = self.make_segmenter(self.baseline)
            await self._set("ready")
            self._after_word()
        elif self._words:
            self._words.pop()
            await self.send(WordsUpdateMsg(words=[w for w, _ in self._words]))
            self._arm_done()

    async def _clear_words(self) -> None:
        self._disarm_done()
        self._done_pending = False
        if self._words:
            self._words = []
            await self.send(WordsUpdateMsg(words=[]))

    def _arm_done(self) -> None:
        self._disarm_done()
        if WORDS_DONE_MS > 0 and self._words and self.by_sign and not self._closed:
            self._done_timer = asyncio.create_task(self._done_after())

    def _disarm_done(self) -> None:
        if self._done_timer and not self._done_timer.done() and self._done_timer is not asyncio.current_task():
            self._done_timer.cancel()
        self._done_timer = None

    async def _done_after(self) -> None:
        await asyncio.sleep(WORDS_DONE_MS / 1000)
        while self._busy() or self.state != "ready":  # a word still being read or picked lands first
            if self.state in ("signing", "calibrating", "listening"):
                return  # another sign started (its word re-arms the timer) or the turn was abandoned
            await asyncio.sleep(0.1)
        self._done_timer = None
        if self._words and self.by_sign and not self.practice:
            self._start_saying()

    def _needs_pick(self, t: list[tuple[str, float]], s: Segment) -> tuple[str | None, list[tuple[str, float]]]:
        """Whether a read goes to the picker, and the options to offer. Reason: None (spoken), "unsure" (below the pick
        threshold), "lookalike" (sure enough, but a look-alike is close behind); "+swap" when a look-alike took the 3rd
        place. Blocking (may run the sign model once more): call it off the event loop."""
        if s.kind == "spell" and self.spell_reader is not None:  # read by the letter reader
            return ("unsure" if t[0][1] < SPELL_PICK_THRESHOLD else None), t
        g1, p1 = t[0]
        if p1 >= self.pick_threshold:
            close = p1 < LOOKALIKE_CAP and any(self._alike_pair(g1, g) and p >= LOOKALIKE_RATIO * p1 for g, p in t[1:3])
            return ("lookalike" if close else None), t
        if len(t) < 3 or not self._alike or any(self._alike_pair(g1, g) for g, _ in t[1:3]):
            return "unsure", t
        shown = {g for g, _ in t[:3]}
        floor = max(LOOKALIKE_MIN_P, LOOKALIKE_SWAP_FLOOR * t[2][1])
        for g, p in self.recognizer.topk(s.context or s.frames, 50):
            if g not in shown and self._alike_pair(g1, g) and p >= floor:
                return "unsure+swap", [*t[:2], (g, p)]
        return "unsure", t

    def _alike_pair(self, a: str, b: str) -> bool:
        return a != b and a in self._alike and self._alike.get(a) == self._alike.get(b)

    def _read_turn(self, ev: TurnEnd) -> list[tuple[Segment, list[tuple[str, float]]]]:
        """Each piece and its guesses, in order: spelled ones by the letter reader; each run of signed ones cut again
        where the sign model is surest (recut.py), or as the segmenter cut them when that finds no cut."""
        from .recut import recut  # recut builds on the segmenter, which imports this module

        around = sorted({f.t: f for f in [*ev.frames, *(f for s in ev.segments for f in s.context or [])]}.values(),
                        key=lambda f: f.t)
        out, run = [], []

        def flush():
            if run:
                frames = [f for f in around if run[0].frames[0].t <= f.t <= run[-1].frames[-1].t]
                spans = [(s.frames[0].t, s.frames[-1].t) for s in run]
                out.extend(recut(frames, around, self.recognizer, spans) or [(s, self._read_segment(s)) for s in run])
                run.clear()

        for s in ev.segments:
            if s.kind == "sign":
                run.append(s)
            else:
                flush()
                out.append((s, self._read_segment(s)))
        flush()
        return out

    def _read_segment(self, s: Segment) -> list[tuple[str, float]]:
        if s.kind == "spell" and self.spell_reader is not None:
            top = self.spell_reader.recognize(s)
            if top and len(top[0][0]) >= SPELL_MIN_LETTERS and top[0][1] >= SPELL_MIN_P:
                return top
            s.kind = "sign"  # a sign with busy fingers, not a spelled word: read, picked and composed as a sign
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
