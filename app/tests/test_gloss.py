import json
import os

import anthropic
import pytest

from server.gloss import Gloss, PlanError, validate_plan
from server.vocab import load_vocab
from test_compose import fake_client

V = load_vocab(extended=None)  # curated only: tests must not depend on the generated extension


def reply(tokens, grammar="statement", tone="neutral"):
    return json.dumps({"tokens": [{"kind": k, "value": v} for k, v in tokens], "grammar": grammar, "tone": tone})


async def test_valid_plan_passes():
    c = fake_client(reply([("sign", "YOU"), ("sign", "WANT"), ("sign", "WHAT")], "wh", "warm"))
    plan = await Gloss(c, V).plan("What would you like?")
    assert [t.gloss for t in plan.tokens] == ["YOU", "WANT", "WHAT"]
    assert plan.grammar == "wh" and plan.tone == "warm"


async def test_unknown_gloss_retried_then_spelled():
    bad = reply([("sign", "ESPRESSO"), ("sign", "WANT")])
    c = fake_client(bad, bad)
    plan = await Gloss(c, V).plan("Want espresso?")
    assert len(c.messages.calls) == 2
    assert plan.tokens[0].kind == "spell" and plan.tokens[0].word == "ESPRESSO"
    assert plan.tokens[1].gloss == "WANT"


async def test_requests_race_fast_and_schema():
    c = fake_client(reply([("sign", "HELLO")]), reply([("sign", "HELLO")]))
    await Gloss(c, V).plan("Hello")
    assert len(c.messages.calls) == 2  # two racing requests, no sequential retry
    assert "format" not in c.messages.calls[0]["output_config"]  # fast: a schema cost 0.5-1.1 s per call
    assert c.messages.calls[1]["output_config"]["format"]["type"] == "json_schema"  # reliable JSON


async def test_chatty_reply_loses_to_schema_racer():
    c = fake_client("I'm ready to translate. Please give me the sentence.", reply([("sign", "HELLO")]))
    plan = await Gloss(c, V).plan("hi")
    assert [t.gloss for t in plan.tokens] == ["HELLO"]  # not the spelled-out fallback


async def test_strict_miss_loses_to_valid_racer():
    c = fake_client(reply([("sign", "HALL")]), reply([("sign", "BATHROOM"), ("sign", "WHERE")], "wh"))
    plan = await Gloss(c, V).plan("Where is the bathroom down the hall?")
    assert [t.gloss for t in plan.tokens] == ["BATHROOM", "WHERE"]  # not the lenient "HALL" spelled out


async def test_malformed_json_falls_back_to_spelling_words():
    c = fake_client("nope", "nope again")
    plan = await Gloss(c, V).plan("Hi Ana!")
    assert [(t.kind, t.word) for t in plan.tokens] == [("spell", "HI"), ("spell", "ANA")]
    assert plan.grammar == "statement" and plan.tone == "neutral"


def test_sad_and_angry_tones_kept_unknown_tone_neutral():
    plan = lambda tone: validate_plan({"tokens": [{"kind": "sign", "value": "I"}, {"kind": "sign", "value": "SAD"}],
                                       "grammar": "statement", "tone": tone}, V, "I'm sad", lenient=False)
    assert plan("sad").tone == "sad" and plan("angry").tone == "angry"
    assert plan("furious").tone == "neutral"


def test_spell_sanitized():
    plan = validate_plan({"tokens": [{"kind": "spell", "value": "Ana-Marie!"}, {"kind": "spell", "value": "!!"}],
                          "grammar": "statement", "tone": "neutral"}, V, "Ana-Marie", lenient=False)
    assert [t.word for t in plan.tokens] == ["ANAMARIE"]


def test_strict_validation_rejects_unknown_gloss():
    with pytest.raises(PlanError):
        validate_plan({"tokens": [{"kind": "sign", "value": "ZEBRA"}], "grammar": "statement", "tone": "neutral"}, V, "x", lenient=False)


async def test_repeated_sign_loses_to_valid_racer():
    c = fake_client(reply([("sign", "NOT"), ("sign", "NOT"), ("sign", "NOT")], "neg"),
                    reply([("sign", "MY"), ("sign", "MOTHER")]))
    plan = await Gloss(c, V).plan("My mother.")
    assert [t.gloss for t in plan.tokens] == ["MY", "MOTHER"]


async def test_repeats_collapsed_when_both_racers_repeat():
    bad = reply([("sign", "MY"), ("sign", "MOTHER"), ("sign", "MOTHER")])
    plan = await Gloss(fake_client(bad, bad), V).plan("My grandmother.")
    assert [t.gloss for t in plan.tokens] == ["MY", "MOTHER"]


def test_repeat_allowed_when_speaker_repeated():
    raw = {"tokens": [{"kind": "sign", "value": "NO"}, {"kind": "sign", "value": "NO"}], "grammar": "neg", "tone": "urgent"}
    assert [t.gloss for t in validate_plan(raw, V, "No, no!", lenient=False).tokens] == ["NO", "NO"]


async def test_english_echoed_for_captions():
    c = fake_client(reply([("sign", "HELLO")]))
    assert (await Gloss(c, V).plan("Hello there")).english == "Hello there"


async def test_fillers_are_dropped_and_the_caption_shows_what_they_meant():
    c = fake_client(json.dumps({"tokens": [{"kind": "sign", "value": "HELLO"}], "grammar": "statement", "tone": "warm",
                                "meant": "Hello!"}))
    plan = await Gloss(c, V).plan("Um, uh, hello, like, hello!")
    assert [t.gloss for t in plan.tokens] == ["HELLO"] and plan.english == "Hello!" and not plan.unclear


async def test_gibberish_is_flagged_not_signed():
    c = fake_client(json.dumps({"tokens": [{"kind": "spell", "value": "BLORP"}], "grammar": "statement", "tone": "neutral",
                                "meant": "", "unclear": True}))
    plan = await Gloss(c, V).plan("blorp fizzle wug")
    assert plan.unclear and plan.tokens == [] and plan.english == "blorp fizzle wug"  # the caption shows what was heard


def test_the_prompt_asks_for_fillers_dropped_and_gibberish_flagged():
    from server.gloss import SCHEMA, SYSTEM
    assert "filler" in SYSTEM and '"unclear": true' in SYSTEM
    assert {"meant", "unclear"} <= set(SCHEMA["properties"]) and {"meant", "unclear"} <= set(SCHEMA["required"])


async def test_system_prompt_lists_vocab():
    c = fake_client(reply([("sign", "HELLO")]))
    await Gloss(c, V).plan("Hello")
    system = c.messages.calls[0]["system"]
    assert "COFFEE — coffee" in system[0]["text"]


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="needs ANTHROPIC_API_KEY")
async def test_live_plan_whats_size():
    plan = await Gloss(anthropic.AsyncAnthropic(max_retries=0), V).plan("What size would you like?")
    assert plan.grammar == "wh"
    assert all(t.gloss in V.glosses for t in plan.tokens if t.kind == "sign")
