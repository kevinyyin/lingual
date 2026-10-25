"""Recognized signs + facial grammar + emotion -> one natural English sentence (Claude)."""

import asyncio
import json
import logging

from .llm import LLMError, ask_json
from .vocab import Vocab
from .voice import EMOTIONS

log = logging.getLogger("asl.compose")

SYSTEM = """You voice a Deaf ASL signer in a live conversation. You get the signs they made (ASL gloss, in signing order), \
any fingerspelled words, and what their face showed. Write the ONE natural, first-person English sentence they mean, \
and pick the emotion it is spoken with.
Rules:
- Keep the ASL meaning, not the ASL word order. Add only the small words English needs (articles, "to", "do").
- question "yn": a yes/no question ending in "?". question "wh": a wh-question ending in "?". Otherwise a statement.
- negation true: the sentence is negated, even if no NOT sign appears.
- Fingerspelled words are names or terms: use them verbatim.
- Never add facts that were not signed. Emotion may color word choice slightly but must not change the meaning.
- The signs come from a camera that sometimes misreads one: if they don't fit together in the order read, write the \
most natural sentence they make together.
- If no sensible sentence can be made from them at all, never invent one: reply {"text": "", "unclear": true}.
- voice: the ONE emotion from this list the sentence is spoken with: """ + ", ".join(EMOTIONS) + """. The face is what \
they feel: stay with it (a sad face is never "excited") and pick the word that best fits it, its intensity (0-1) and the \
sentence, e.g. angry 0.4 "frustrated", angry 0.9 "outraged", sad with "I miss her" "melancholic". A neutral face stays \
"neutral" unless the words carry a feeling of their own ("thank you": "grateful", "sorry": "apologetic").
Reply as JSON: {"text": "<sentence>", "voice": "<emotion>"}"""


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

    async def compose(self, signs, spelled, grammar, emotion) -> tuple[str, dict]:
        """The sentence, and the emotion with the voice's pick from Cartesia's list added (the face label's own when
        there is no valid pick). "" when the signs make no sentence (a misread turn), which the session shows instead
        of saying."""
        user = json.dumps({"signs": signs, "fingerspelled": spelled, "question": grammar["question"],
                           "negation": grammar["negation"], "emotion": emotion["label"],
                           "intensity": round(emotion["intensity"], 2)})
        # two identical requests race: latency spikes are per request (measured pairs 0.7-1.4 s, the faster never over 0.9 s)
        tasks = [asyncio.create_task(self._ask(user, n)) for n in (1, 2)]
        try:
            for done in asyncio.as_completed(tasks):
                if (out := await done) is not None:  # "" text = the signs make no sentence
                    text, voice = out
                    return text, emotion | ({"voice": voice} if voice else {})
        finally:
            for t in tasks:
                t.cancel()
        return self._fallback(signs, spelled), emotion

    async def _ask(self, user: str, attempt: int) -> tuple[str, str | None] | None:
        try:
            # no output schema: it added 0.5-1.1 s per call (measured); the prompt asks for JSON and ask_json checks it
            out = await ask_json(self.client, SYSTEM, user, None, max_tokens=200)
            if out.get("unclear") is True:
                return "", None
            text = str(out.get("text", "")).strip()
            if text:
                voice = out.get("voice")
                if voice not in EMOTIONS:
                    log.warning("compose attempt %d: %r is not a Cartesia emotion", attempt, voice)
                    voice = None
                return text, voice
            log.warning("compose attempt %d: empty text", attempt)
        except LLMError as e:
            log.warning("compose attempt %d failed: %s", attempt, e)
        return None
