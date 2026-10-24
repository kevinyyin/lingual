import json

import pytest

from server.protocol import FrameMsg, SignPlanMsg, SignToken, parse_client


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
