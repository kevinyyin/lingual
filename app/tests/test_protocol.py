import json
import re
from pathlib import Path

import pytest

from typing import get_args

from server.protocol import Buddy, FrameMsg, RecognizedMsg, SignPlanMsg, SignToken, parse_client
from server.voice import BUDDY_VOICES


def _frame(**over):
    f = {
        "type": "frame",
        "t": 12.5,
        "left": None,
        "right": [[0.1, 0.2, 0.0]] * 21,
        "pose": [[0.5, 0.5, 0.0, 1.0]] * 33,
        "blend": {"browInnerUp": 0.1},
        "head": [0.0, 0.0, 0.0],
    }
    f.update(over)
    return json.dumps(f)


def test_parse_frame_with_missing_hand():
    msg = parse_client(_frame())
    assert isinstance(msg, FrameMsg)
    assert msg.left is None
    assert len(msg.right) == 21


def test_parse_rejects_unknown_type():
    with pytest.raises(ValueError):
        parse_client('{"type":"nope"}')


def test_sign_plan_roundtrip():
    plan = SignPlanMsg(
        tokens=[SignToken(kind="sign", gloss="COFFEE"), SignToken(kind="spell", word="ANA")],
        grammar="yn",
        tone="warm",
        english="Coffee, Ana?",
    )
    assert SignPlanMsg.model_validate_json(plan.model_dump_json()) == plan


def test_say_text_capped():
    assert parse_client(json.dumps({"type": "say", "text": "a" * 200})).text == "a" * 200
    with pytest.raises(ValueError):
        parse_client(json.dumps({"type": "say", "text": "a" * 201}))
    with pytest.raises(ValueError):
        parse_client(json.dumps({"type": "say", "text": ""}))


def test_buddy_is_one_of_the_six_with_a_voice():
    assert set(get_args(Buddy)) == set(BUDDY_VOICES) and len(BUDDY_VOICES) == 6
    assert parse_client(json.dumps({"type": "buddy", "buddy": "lexi"})).buddy == "lexi"
    with pytest.raises(ValueError):  # the page names a buddy, never a voice id
        parse_client(json.dumps({"type": "buddy", "buddy": "0834f3df-e650-4766-a20c-5a93a43aa6e3"}))


def test_every_buddy_on_the_page_has_a_voice_and_an_avatar_file():
    web = Path(__file__).resolve().parent.parent / "web"
    ids = re.findall(r'id: "(\w+)"', (web / "src" / "buddies.ts").read_text())
    assert ids == list(BUDDY_VOICES)
    assert all((web / "public" / "avatar" / "toon" / f"{b}.glb").exists() for b in ids)
