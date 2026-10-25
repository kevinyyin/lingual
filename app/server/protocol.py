"""Browser <-> backend WebSocket messages. Mirrored by web/src/ws.ts."""

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

Point3 = list[float]  # [x, y, z]
Point4 = list[float]  # [x, y, z, visibility]


class Frame(BaseModel):
    t: float  # ms, performance.now()
    left: list[Point3] | None = None  # 21 hand landmarks
    right: list[Point3] | None = None
    pose: list[Point4]  # 33 pose landmarks
    blend: dict[str, float] | None = None  # 52 ARKit-named face blendshapes
    head: tuple[float, float, float] | None = None  # yaw, pitch, roll (degrees)
    face: list[list[float]] | None = None  # 6 face-mesh points [x, y]: nose tip, lips, brows, chin (spike_classifier.py)
    mesh: list[Point3] | None = None  # the 468 face-mesh points, while a hand is up: the fingerspelling reader's


# --- browser -> backend ---

class FrameMsg(Frame):
    type: Literal["frame"] = "frame"


class CalibrateMsg(BaseModel):
    type: Literal["calibrate"] = "calibrate"


class PickMsg(BaseModel):
    type: Literal["pick"] = "pick"
    segment: int
    gloss: str


class AvatarDoneMsg(BaseModel):
    type: Literal["avatar_done"] = "avatar_done"


class PracticeMsg(BaseModel):
    """Teaching mode (teach.html): sign turns are graded by the page, not composed or spoken; speech is ignored."""
    type: Literal["practice"] = "practice"
    on: bool
    signs: list[str] = []  # the course's signs: each turn is read as one of these (empty: any sign)


class ModeMsg(BaseModel):
    """Translator's switch: voice (the hearing person talks, the avatar signs; the camera is ignored) or signing (the
    camera reads signs and they are said aloud; speech is ignored). Until a page sends one, both directions run."""
    type: Literal["mode"] = "mode"
    mode: Literal["voice", "signing"]
    demo: bool = False  # the page was opened with ?demo=1: signed turns snap to the demo sentences (demo_sentences.py)
    by_sign: bool = False  # sign by sign: each turn is one word, shown as it is read; the sentence is said on Done


class TalkMsg(BaseModel):
    """Translator's push to talk: speech is heard only while Space or the Hold to talk button is held (it heard every
    word said near the mic)."""
    type: Literal["talk"] = "talk"
    on: bool


class WordsMsg(BaseModel):
    """Sign by sign: undo drops the last word read, done says the words as a sentence, clear starts over."""
    type: Literal["words"] = "words"
    action: Literal["undo", "done", "clear"]


class RedoMsg(BaseModel):
    """Teaching mode: the learner tapped the frozen camera to sign again: the attempt still being read is dropped."""
    type: Literal["redo"] = "redo"


class SayMsg(BaseModel):
    """Teaching mode: a short line for the plushie to say (greeting, praise, "your turn"). Capped: it goes to TTS."""
    type: Literal["say"] = "say"
    text: str = Field(min_length=1, max_length=200)


class AnswerMsg(BaseModel):
    """Teaching mode: the child answered, so the plushie shows a check (and flaps), an X, or a ! for "try again"
    (server/plush.py)."""
    type: Literal["answer"] = "answer"
    result: Literal["right", "wrong", "retry"]


Buddy = Literal["leo", "benji", "zeke", "lexi", "avery", "daisy"]


class BuddyMsg(BaseModel):
    """The signing buddy picked on the Profile tab: its voice (server/voice.py BUDDY_VOICES) says every later line."""
    type: Literal["buddy"] = "buddy"
    buddy: Buddy


class MirrorMsg(BaseModel):
    """The avatar mirror (?mode=mirror): each sign of a turn is read and labelled on the page (RecognizedMsg with
    spans); nothing is composed, said or picked, and speech is ignored."""
    type: Literal["mirror"] = "mirror"
    on: bool


class RecordMsg(BaseModel):
    type: Literal["record"] = "record"
    signer: str
    gloss: str
    frames: list[Frame]


ClientMsg = Annotated[
    Union[FrameMsg, CalibrateMsg, PickMsg, AvatarDoneMsg, PracticeMsg, ModeMsg, TalkMsg, SayMsg, AnswerMsg, RedoMsg, BuddyMsg,
          MirrorMsg, RecordMsg, WordsMsg],
    Field(discriminator="type"),
]
_client = TypeAdapter(ClientMsg)


def parse_client(raw: str | bytes) -> ClientMsg:
    try:
        return _client.validate_json(raw)
    except ValidationError as e:
        raise ValueError(str(e)) from e


# --- backend -> browser ---

# reading: the signer's hands came down and the turn is being read (frames are ignored until it is done)
State = Literal["calibrating", "ready", "signing", "reading", "picking", "speaking", "listening", "avatar"]


class StateMsg(BaseModel):
    type: Literal["state"] = "state"
    state: State


class PickOption(BaseModel):
    gloss: str
    english: str
    p: float


class PickRequestMsg(BaseModel):
    type: Literal["pick_request"] = "pick_request"
    segment: int
    options: list[PickOption]


class YouSaidMsg(BaseModel):
    type: Literal["you_said"] = "you_said"
    text: str
    emotion: str
    unclear: bool = False  # the signs made no sentence: nothing was said aloud (text is empty)
    read: list["SignToken"] = []  # what the camera read, in order: the mini avatar signs it back


class CaptionMsg(BaseModel):
    type: Literal["caption"] = "caption"
    text: str
    final: bool


class SignToken(BaseModel):
    kind: Literal["sign", "spell"]
    gloss: str | None = None
    word: str | None = None


class SignPlanMsg(BaseModel):
    type: Literal["sign_plan"] = "sign_plan"
    tokens: list[SignToken]
    grammar: Literal["yn", "wh", "neg", "statement"]
    tone: Literal["warm", "apologetic", "urgent", "neutral"]
    english: str
    unclear: bool = False  # the words made no sense: nothing is signed, and english is what was heard


class RecognizedMsg(BaseModel):
    """Practice-mode result: the top guesses for each read of the turn (the whole turn, then its last moments alone; the
    page counts the sign in either), best first. Mirror mode: one entry per sign in signing order, and `spans` holds
    each sign's first and last frame time (the page's Frame.t)."""
    type: Literal["recognized"] = "recognized"
    segments: list[list[PickOption]]
    spans: list[tuple[float, float]] = []


class WordsUpdateMsg(BaseModel):
    """Sign by sign: the words read so far, in order (the page shows them as chips)."""
    type: Literal["words_update"] = "words_update"
    words: list[SignToken]


class RecordedMsg(BaseModel):
    type: Literal["recorded"] = "recorded"
    ok: bool
    detail: str  # saved path or error


ServerMsg = Union[StateMsg, PickRequestMsg, YouSaidMsg, CaptionMsg, SignPlanMsg, RecognizedMsg, RecordedMsg, WordsUpdateMsg]
