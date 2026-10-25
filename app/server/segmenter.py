"""Motion-energy segmentation: sign/fingerspell segments inside a turn, and end of turn (hands back to rest).

Energy per frame = wrist speed + fingertip-spread change, in shoulder widths per 30 fps frame interval, lightly
smoothed. Durations are in ms, not frames: the browser delivers 12-15 frames/s (MediaPipe's cost), not the 30 the
thresholds were set at, which doubled the end-of-turn wait and dropped short signs.
"""

from collections import deque

import numpy as np

from .protocol import Frame
from .session import Baseline, Segment, SegEvent, TurnEnd, TurnStart

FRAME_MS = 1000.0 / 30  # energy unit
REST_MS = 450.0  # hands below the rest line this long ends the turn (15 frames at 30 fps)
# low energy this long splits segments. 90 ms cut live signs at their holds (HELLO at the forehead read as five signs);
# training.eval_live, sign spoken without a pick, curated signs: 90 ms 0.381, 200 0.395, 300 0.406, 450 0.415 (450 merges
# the brief pauses between signs in a sentence, which one-sign videos can't show)
PAUSE_MS = 300.0
PAUSE_K = 1.5  # pause threshold = max(motion_floor * PAUSE_K, MIN_ENERGY)
MIN_ENERGY = 0.02  # shoulder widths per 30 fps frame
MIN_SEG_MS = 160.0  # shorter segments are noise (6 frames at 30 fps)
# ...unless the turn has no other: then its longest shorter piece of LONE_MIN_FRAMES frames over LONE_MIN_MS is its one
# segment (a quick sign at 15 fps never reaches 160 ms). 3 frames lost BABY in the teaching clips (lessons 317 -> 315,
# 2 more turns with no sign); 2 frames and 60 ms means 2 frames at 15 fps (67 ms) but 3 at 30 fps, never a lone frame.
LONE_MIN_FRAMES, LONE_MIN_MS = 2, 60.0
# Fingerspelling: one hand, the wrist staying put, the fingers busier than it. Real spelling (Kaggle, 185 phrases) vs
# WLASL sign pieces (1,444): box 0.6, ratio 2.0 and never a second hand caught 32% of spelling (0.7% of signs); these
# catch 95% (3.5%, which the letter reader then hands back to the sign model: session._read_segment).
SPELL_BOX = 0.7  # wrist stays inside this many shoulder widths
SPELL_RATIO = 1.1  # fingertip motion / wrist motion above this = fingerspelling
SPELL_TWO_HANDS = 0.2  # the other hand in at most this share of frames (the resting hand drifts into the picture)
# Context handed to the sign recognizer around each segment, never past halfway to a neighbouring sign. ASL Citizen
# test signers, ST-GCN top-1 by margin around the hands-visible span: 150 ms 0.49, 300 ms 0.57, 500 ms 0.60, whole clip
# 0.61 (training/eval_recognition.py cache).
CONTEXT_MS = 500.0
PRE_ROLL = 30  # frames kept from before the turn starts (the hands rising), >= CONTEXT_MS at any frame rate
# The whole turn (a lesson's read) keeps this much of each still stretch. The sign model learned from recordings flowing
# from rest straight into the sign; a learner raising the hands and holding in position first read worse (course
# replays, 1 s hold: 87% pass vs 92%). Cut to 200 ms: 93%, recordings without a hold unchanged, false passes the same.
HOLD_KEEP_MS = 200.0
# After the last sign's hold the hands going back down to rest are cut as a piece of their own (29% of one-sign turns),
# read as a junk word. A last piece moving mostly down (DOWN_FRAC of the moving wrist's path) and fast (under DOWN_MS)
# is that: single-sign WLASL replays drop 112 of 141 such pieces, and 10 of 371 real signs ending a sentence.
DOWN_FRAC = 0.8
DOWN_MS = 700.0
TIPS = [4, 8, 12, 16, 20]


def _width(f: Frame) -> float:
    a, b = np.asarray(f.pose[11][:2]), np.asarray(f.pose[12][:2])
    return max(float(np.linalg.norm(a - b)), 1e-3)


def _wrists(f: Frame) -> np.ndarray:
    return np.asarray([f.pose[15][:2], f.pose[16][:2]], dtype=np.float32)


def _tips(hand) -> np.ndarray:
    h = np.asarray(hand, dtype=np.float32)[:, :2]
    return h[TIPS] - h[0]


def _finger_motion(a: Frame, b: Frame) -> float:
    d = [np.linalg.norm(_tips(getattr(b, s)) - _tips(getattr(a, s)), axis=1).mean()
         for s in ("left", "right") if getattr(a, s) is not None and getattr(b, s) is not None]
    return float(max(d)) if d else 0.0


class Segmenter:
    def __init__(self, baseline: Baseline):
        self.b = baseline
        self.rest_ms = REST_MS  # the avatar mirror waits longer (session.MIRROR_REST_MS)
        self.thresh = max(baseline.motion_floor * PAUSE_K, MIN_ENERGY)
        self.turn: list[Frame] | None = None
        self.segments: list[Segment] = []
        self.cur: list[Frame] | None = None
        self.low = 0  # trailing low-energy frames in the open segment
        self.low_t: float | None = None  # when they began
        self.still_t: float | None = None  # when the hands went below the rest line
        self.energy = 0.0
        self.prev: Frame | None = None
        self.before: deque[Frame] = deque(maxlen=PRE_ROLL)  # frames outside a turn
        self.pre: list[Frame] = []  # the pre-roll of the current turn
        self.still: set[float] = set()  # times of the turn's frames with too little energy to be signing
        self.short: list[Frame] | None = None  # the turn's longest piece under MIN_SEG_MS (LONE_MIN_MS)

    def _raised(self, f: Frame) -> bool:
        return bool((_wrists(f)[:, 1] < self.b.rest_y).any())

    def push(self, f: Frame) -> SegEvent | None:
        raised = self._raised(f)
        # No hand in the tracker's sight: they are down (below a laptop's picture), whatever the pose model guesses for
        # the wrists. Its guesses wander off-screen (y 1.05-1.15 against a rest line learned from guesses at 1.24-1.38,
        # 2026-10-24), which kept turns from ever ending. Mid-sign, a hand the tracker loses for a moment is a pause.
        unseen = f.left is None and f.right is None
        if self.turn is None:
            if not raised or unseen:
                self.before.append(f)
                return None
            self.turn, self.segments, self.cur, self.low, self.energy, self.prev = [f], [], None, 0, 0.0, f
            self.low_t = self.still_t = None
            self.pre = list(self.before)
            self.still = set()
            self.short = None
            self.before.clear()
            return TurnStart()

        self.turn.append(f)
        w = _width(f)
        raw = (np.linalg.norm(_wrists(f) - _wrists(self.prev), axis=1).max() + _finger_motion(self.prev, f)) / w
        dt = f.t - self.prev.t
        raw *= FRAME_MS / dt if dt > 0 else 1.0
        self.energy = 0.5 * self.energy + 0.5 * float(raw)
        self.prev = f
        if self.energy < self.thresh:
            self.still.add(f.t)

        if not raised or unseen:
            if self.still_t is None:
                self.still_t = f.t
            if not raised and self.cur is not None:
                self._close()  # hands down ends the sign in progress
            if f.t - self.still_t >= self.rest_ms:
                return self._end()
            if raised:
                self._idle_frame(f)
            return None
        self.still_t = None
        if self.energy >= self.thresh:
            if self.cur is None:
                self.cur = []
            self.cur.append(f)
            self.low, self.low_t = 0, None
        else:
            self._idle_frame(f)
        return None

    def _idle_frame(self, f: Frame) -> None:
        if self.cur is None:
            return
        self.cur.append(f)
        self.low += 1
        if self.low_t is None:
            self.low_t = f.t
        if f.t - self.low_t >= PAUSE_MS:
            self._close()

    def _close(self) -> None:
        frames = self.cur[: len(self.cur) - self.low] if self.cur else []
        self.cur, self.low, self.low_t = None, 0, None
        if frames and frames[-1].t - frames[0].t >= MIN_SEG_MS:
            self.segments.append(Segment(frames=frames, kind=_kind(frames)))
        elif len(frames) >= LONE_MIN_FRAMES and frames[-1].t - frames[0].t >= LONE_MIN_MS and (self.short is None or len(frames) > len(self.short)):
            self.short = frames

    def _end(self) -> TurnEnd:
        self._close()
        if not self.segments and self.short:
            self.segments.append(Segment(frames=self.short, kind=_kind(self.short)))
        self.short = None
        turn, self.turn = self.turn, None
        rest_ms = turn[-1].t - self.still_t
        frames = self.pre + turn
        end = self.segments[-1].frames[-1].t if self.segments else 0.0  # the whole turn still includes the settling
        if len(self.segments) > 1 and _to_rest(self.segments[-1]):
            self.segments.pop()
        _add_context(self.segments, frames)
        whole = []  # TurnEnd.whole: every frame from the first sign's context to the last's, holds between included
        if self.segments:
            lo, hi = self.segments[0].context[0].t, end + CONTEXT_MS
            whole = _cut_holds([f for f in frames if lo <= f.t <= hi], self.still)
        merged: list[Segment] = []
        for s in self.segments:  # letters split by tiny holds belong to one spelled word
            if merged and s.kind == merged[-1].kind == "spell":
                merged[-1] = Segment(frames=merged[-1].frames + s.frames, kind="spell")
            else:
                merged.append(s)
        return TurnEnd(segments=merged, frames=turn, rest_ms=rest_ms, whole=whole)


def _cut_holds(frames: list[Frame], still: set[float]) -> list[Frame]:
    """Each still stretch cut to its first HOLD_KEEP_MS; later frames' times moved up to close the gap (the sign model
    resamples by time, and would fill a gap with the hold again)."""
    out, since, shift = [], None, 0.0
    for prev, f in zip([None, *frames], frames):
        if f.t in still:
            since = f.t if since is None else since
            if f.t - since > HOLD_KEEP_MS:
                shift += f.t - prev.t
                continue
        else:
            since = None
        out.append(f.model_copy(update={"t": f.t - shift}) if shift else f)
    return out


def _to_rest(s: Segment) -> bool:
    """The hands going back down to rest, not a sign (DOWN_FRAC, DOWN_MS). A spelled word's wrist barely moves, so
    any drift could look downward: never one of those."""
    if s.kind != "sign" or s.frames[-1].t - s.frames[0].t >= DOWN_MS:
        return False
    steps = np.diff(np.stack([_wrists(f) for f in s.frames]), axis=0)  # (frames - 1, wrist, xy)
    path = np.linalg.norm(steps, axis=2).sum(0)
    k = int(np.argmax(path))  # the wrist that moved most
    return path[k] > 0 and steps[:, k, 1].clip(min=0).sum() >= DOWN_FRAC * path[k]  # image y grows downward


def _add_context(segments: list[Segment], frames: list[Frame]) -> None:
    for k, s in enumerate(segments):
        start, end = s.frames[0].t, s.frames[-1].t
        lo, hi = start - CONTEXT_MS, end + CONTEXT_MS
        if k > 0:
            lo = max(lo, (segments[k - 1].frames[-1].t + start) / 2)
        if k + 1 < len(segments):
            hi = min(hi, (end + segments[k + 1].frames[0].t) / 2)
        s.context = [f for f in frames if lo <= f.t <= hi]


def _kind(frames: list[Frame]) -> str:
    with_hand = [f for f in frames if f.left is not None or f.right is not None]
    if not with_hand or np.mean([f.left is not None and f.right is not None for f in with_hand]) > SPELL_TWO_HANDS:
        return "sign"
    w = np.median([_width(f) for f in frames])
    side = "right" if sum(f.right is not None for f in with_hand) >= len(with_hand) / 2 else "left"
    hand = [f for f in with_hand if getattr(f, side) is not None]
    wrist = np.asarray([getattr(f, side)[0][:2] for f in hand])
    box = (wrist.max(0) - wrist.min(0)).max() / w
    wrist_motion = np.linalg.norm(np.diff(wrist, axis=0), axis=1).mean() / w if len(hand) > 1 else 0.0
    finger = np.mean([_finger_motion(a, b) for a, b in zip(hand, hand[1:])]) / w if len(hand) > 1 else 0.0
    if box < SPELL_BOX and finger > SPELL_RATIO * max(wrist_motion, 1e-3):
        return "spell"
    return "sign"
