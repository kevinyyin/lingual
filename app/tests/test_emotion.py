import math

from frames import make_frame
from server.emotion import read_emotion, signing_stats
from server.readers import FaceReaders
from server.session import Baseline


def base():
    speed, size = signing_stats(turn({}, amp=0.1))  # the signer's usual signing
    return Baseline(rest_y=0.8, brow={}, motion_floor=0.0, speed=speed, size=size)


def turn(blend, amp=0.1, n=30, face=True):
    """Right wrist circles with amplitude `amp` (bigger amp = faster and larger signing)."""
    out = []
    for i in range(n):
        f = make_frame(t=i * 33.0)
        f.pose[16] = [0.4 + amp * math.cos(i / 2), 0.5 + amp * math.sin(i / 2), 0.0, 1.0]
        f.blend = dict(blend) if face else None
        out.append(f)
    return out


def test_smile_is_happy():
    e = read_emotion(turn({"mouthSmileLeft": 0.7, "mouthSmileRight": 0.7, "cheekSquintLeft": 0.4, "cheekSquintRight": 0.4}), base())
    assert e["label"] == "happy" and e["intensity"] > 0.5


def test_frown_slow_small_is_sad():
    e = read_emotion(turn({"mouthFrownLeft": 0.5, "mouthFrownRight": 0.5}, amp=0.03), base())
    assert e["label"] == "sad"


def test_sneer_fast_big_is_angry():
    e = read_emotion(turn({"noseSneerLeft": 0.5, "noseSneerRight": 0.5, "mouthPressLeft": 0.4, "mouthPressRight": 0.4}, amp=0.25), base())
    assert e["label"] == "angry"


def test_fast_big_signing_raises_intensity():
    face = {"noseSneerLeft": 0.3, "noseSneerRight": 0.3}
    calm = read_emotion(turn(face, amp=0.05), base())
    big = read_emotion(turn(face, amp=0.3), base())
    assert big["intensity"] > calm["intensity"]


def test_plain_face_is_neutral():
    assert read_emotion(turn({"mouthSmileLeft": 0.05}), base())["label"] == "neutral"


def test_a_slight_frown_is_sad():
    """MediaPipe's frown barely moves: a turn averaging 0.1 is a clear frown, not noise."""
    assert read_emotion(turn({"mouthFrownLeft": 0.1, "mouthFrownRight": 0.1}, amp=0.05), base())["label"] == "sad"


def test_a_slight_sneer_is_angry():
    assert read_emotion(turn({"noseSneerLeft": 0.12, "noseSneerRight": 0.12}), base())["label"] == "angry"


def test_resting_pucker_and_squint_are_neutral():
    """The median logged face: pucker 0.67, eyes squinted 0.23; both once made half of sad and angry."""
    face = {"mouthPucker": 0.67, "eyeSquintLeft": 0.23, "eyeSquintRight": 0.23, "mouthPressLeft": 0.01}
    assert read_emotion(turn(face), base())["label"] == "neutral"


def test_pressed_lips_alone_are_neutral():
    """The mouth morpheme MM presses the lips with no feeling."""
    assert read_emotion(turn({"mouthPressLeft": 0.5, "mouthPressRight": 0.5}), base())["label"] == "neutral"


def test_the_face_is_read_above_the_resting_face():
    b = base()
    b.face = {"mouthFrownLeft": 0.2, "mouthFrownRight": 0.2}  # this signer's mouth corners rest low
    assert read_emotion(turn({"mouthFrownLeft": 0.2, "mouthFrownRight": 0.2}), b)["label"] == "neutral"
    assert read_emotion(turn({"mouthFrownLeft": 0.35, "mouthFrownRight": 0.35}), b)["label"] == "sad"


def test_brows_ignored():
    face = {"mouthSmileLeft": 0.4, "mouthSmileRight": 0.4}
    brows = {k: 1.0 for k in ("browInnerUp", "browOuterUpLeft", "browOuterUpRight", "browDownLeft", "browDownRight")}
    assert read_emotion(turn(face), base()) == read_emotion(turn({**face, **brows}), base())


def test_no_face_neutral():
    assert read_emotion(turn({}, face=False), base()) == {"label": "neutral", "intensity": 0.0}


def test_readers_fill_signing_baseline_from_first_turn():
    b = Baseline(rest_y=0.8, brow={}, motion_floor=0.0)
    grammar, emotion = FaceReaders().read(turn({"mouthSmileLeft": 0.7, "mouthSmileRight": 0.7}), b)
    assert b.speed and b.size
    assert grammar == {"question": None, "negation": False} and emotion["label"] == "happy"
