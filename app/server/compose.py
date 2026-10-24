"""Recognized signs + facial grammar + emotion -> one natural English sentence (Claude)."""

import asyncio
import json
import logging

from .llm import LLMError, ask_json
from .vocab import Vocab

log = logging.getLogger("asl.compose")

SYSTEM = """You voice a Deaf ASL signer in a live conversation. You get the signs they made (ASL gloss, in signing order), \
any fingerspelled words, and what their face showed. Write the ONE natural, first-person English sentence they mean.
Rules:
- Keep the ASL meaning, not the ASL word order. Add only the small words English needs (articles, "to", "do").
- question "yn": a yes/no question ending in "?". question "wh": a wh-question ending in "?". Otherwise a statement.
- negation true: the sentence is negated, even if no NOT sign appears.
- Fingerspelled words are names or terms: use them verbatim.
- Never add facts that were not signed. Emotion may color word choice slightly but must not change the meaning.
Reply as JSON: {"text": "<sentence>"}"""


class Composer:
    def __init__(self, client, vocab: Vocab):
        self.client, self.vocab = client, vocab

    def _fallback(self, signs: list[str], spelled: list[str]) -> str:
        words = [s.english if (s := self.vocab.get(g)) else g.lower() for g in signs]
        return " ".join(words + spelled)

    async def warm(self) -> None:
        """Cheapest real call on the compose path, so the first turn doesn't pay connection and cold-start cost."""
        try:
            await ask_json(self.client, SYSTEM, "{}", None, max_tokens=1)
        except LLMError:
            pass  # a 1-token reply never parses; only the round trip matters

    async def compose(self, signs, spelled, grammar, emotion) -> str:
        user = json.dumps({"signs": signs, "fingerspelled": spelled, "question": grammar["question"],
                           "negation": grammar["negation"], "emotion": emotion["label"],
                           "intensity": round(emotion["intensity"], 2)})
        # two identical requests race: latency spikes are per request (measured pairs 0.7-1.4 s, the faster never over 0.9 s)
        tasks = [asyncio.create_task(self._ask(user, n)) for n in (1, 2)]
        try:
            for done in asyncio.as_completed(tasks):
                if text := await done:
                    return text
        finally:
            for t in tasks:
                t.cancel()
        return self._fallback(signs, spelled)

    async def _ask(self, user: str, attempt: int) -> str | None:
        try:
            # no output schema: it added 0.5-1.1 s per call (measured); the prompt asks for JSON and ask_json checks it
            text = str((await ask_json(self.client, SYSTEM, user, None, max_tokens=200)).get("text", "")).strip()
            if text:
                return text
            log.warning("compose attempt %d: empty text", attempt)
        except LLMError as e:
            log.warning("compose attempt %d failed: %s", attempt, e)
        return None
