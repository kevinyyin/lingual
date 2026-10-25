import json
import os
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from server.compose import Composer
from server.vocab import load_vocab

V = load_vocab(extended=None)  # curated only: tests must not depend on the generated extension
NEUTRAL = {"label": "neutral", "intensity": 0.0}
YN = {"question": "yn", "negation": False}


class FakeMessages:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    async def create(self, **kw):
        self.calls.append(kw)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=r)])


def fake_client(*replies):
    return SimpleNamespace(messages=FakeMessages(replies))


async def test_yn_question_passes_grammar_to_prompt():
    c = fake_client(json.dumps({"text": "Do you want coffee?"}))
    text = await Composer(c, V).compose(["COFFEE", "WANT"], [], YN, NEUTRAL)
    assert text == "Do you want coffee?"
    user = c.messages.calls[0]["messages"][0]["content"]
    assert '"question": "yn"' in user and "COFFEE" in user
    assert c.messages.calls[0]["model"] == "claude-haiku-5-5"
    assert "format" not in c.messages.calls[0]["output_config"]  # a schema cost 0.5-1.1 s per call


async def test_bad_json_both_attempts_falls_back():
    c = fake_client("oops", "still not json")
    assert await Composer(c, V).compose(["COFFEE", "WANT"], [], YN, NEUTRAL) == "coffee want"
    assert len(c.messages.calls) == 2


async def test_timeout_falls_back():
    err = anthropic.APITimeoutError(request=httpx2.Request("POST", "https://api.anthropic.com"))
    c = fake_client(err, err)
    assert await Composer(c, V).compose(["COFFEE"], ["ANA"], YN, NEUTRAL) == "coffee ANA"


async def test_empty_text_falls_back():
    c = fake_client(json.dumps({"text": "  "}), json.dumps({"text": ""}))
    assert await Composer(c, V).compose(["HELLO"], [], YN, NEUTRAL) == "hello"


async def test_signs_that_make_no_sentence_come_back_empty():
    c = fake_client(json.dumps({"text": "", "unclear": True}))
    assert await Composer(c, V).compose(["PIZZA", "WHO", "SICK"], [], YN, NEUTRAL) == ""


def test_the_prompt_asks_to_flag_signs_that_make_no_sentence():
    from server.compose import SYSTEM
    assert '"unclear": true' in SYSTEM


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="needs ANTHROPIC_API_KEY")
async def test_live_compose():
    text = await Composer(anthropic.AsyncAnthropic(max_retries=0), V).compose(["COFFEE", "WANT"], [], YN, NEUTRAL)
    assert text.endswith("?") and "coffee" in text.lower()


async def test_missing_credentials_falls_back():
    c = fake_client(TypeError("Could not resolve authentication method"), TypeError("again"))
    assert await Composer(c, V).compose(["HELLO"], [], YN, NEUTRAL) == "hello"


async def test_two_requests_race_and_the_fast_one_wins():
    """API latency spikes are per request, so a slow reply must not hold up the turn."""
    import asyncio

    delays = [1.0, 0.01]

    class Racing:
        calls = 0

        async def create(self, **kw):
            Racing.calls += 1
            await asyncio.sleep(delays.pop(0))
            return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=json.dumps({"text": "Hi."}))])

    async with asyncio.timeout(0.5):
        assert await Composer(SimpleNamespace(messages=Racing()), V).compose(["HELLO"], [], YN, NEUTRAL) == "Hi."
    assert Racing.calls == 2


async def test_bad_reply_waits_for_the_other():
    c = fake_client("oops", json.dumps({"text": "Hello."}))
    assert await Composer(c, V).compose(["HELLO"], [], YN, NEUTRAL) == "Hello."


async def test_warm_is_one_tiny_call_and_never_raises():
    c = fake_client(TypeError("no credentials"))
    await Composer(c, V).warm()
    assert len(c.messages.calls) == 1 and c.messages.calls[0]["max_tokens"] == 1
