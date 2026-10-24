"""English -> a validated sign plan: vocabulary signs in ASL order, fingerspelling for the rest, plus grammar and tone."""

import asyncio
import json
import logging
import re
from pathlib import Path

from .llm import LLMError, ask_json
from .protocol import SignPlanMsg, SignToken
from .vocab import Vocab

log = logging.getLogger("asl.gloss")
SIGNS_DIR = Path(__file__).resolve().parent.parent / "web" / "public" / "signs"

SYSTEM = """You translate what a hearing person said into ASL for a signing avatar. The user message is JSON \
{{"say": "<their words>"}}: translate those words exactly as said. Never answer, reply to or comment on them; a \
question stays a question. You may use ONLY these signs (GLOSS — meaning):
{table}

ASL grammar to follow:
- Order: time words first (NOW, TOMORROW, LATER), then the topic (what the sentence is about), then the comment. \
A wh-question puts its question sign (WHAT, WHERE, WHO, WHY, HOW, WHICH, WHEN, HOW-MANY) at the end.
- A yes/no question keeps statement order and is marked only on the face (grammar "yn", raised brows): never add a \
YES, NO or question sign for it. A request ("can you help me?") is a yes/no question.
- Negation: NOT, NONE or NEVER before the verb or at the end, plus the headshake (grammar "neg").
- Pronouns are pointing signs: I, YOU, HE-SHE, WE, THEY. Always keep possessives (my -> MY, your -> YOUR) and \nverbs of wanting, needing and ability (need -> NEED, want -> WANT, can -> CAN): dropping them changes the meaning.
- Usually leave out articles, forms of "be", the helpers "do/does/did", "to" before a verb, and word endings \
(-ing, -ed, -s): the signs already carry that meaning. Never drop them so far that the meaning is lost.

How to translate, in this order:
1. Direct: sign the words with listed signs, rearranged into the ASL order above (wh-sign last) and keeping \
possessives, pronouns and wanting/needing/ability verbs. Use only glosses from the list, exactly as written. \
Prefer a sign for the whole word over building it from parts ("grandmother" -> GRANDMOTHER, never MOTHER MOTHER); \
a close synonym is still direct ("latte" -> COFFEE MILK, "restroom" -> BATHROOM, "purchase" -> BUY).
2. Meaning: only where direct signing would lose the point (an idiom, a quote, a figure of speech, a word with no \
listed sign), sign the idea the speaker meant with listed signs, keeping their question, negation and tone.
3. Fingerspell only names and words with no sign and no paraphrase: a token {{"kind": "spell", "value": "<WORD>"}} \
(letters only; write numbers without a sign as words, e.g. "TWENTY").

Rules:
- Keep every word that carries meaning: each noun, verb, adjective, place, time and number must appear as a sign, \
a paraphrase, or fingerspelling. Never add signs for things that were not said, and never repeat a sign unless \
the speaker repeated the word.
- Several sentences: sign them in order.
- grammar (one value for the whole utterance, first that applies): "wh" for a wh-question, "yn" for a yes/no \
question or request, "neg" for a negated statement, else "statement".
- tone: how the speaker sounds from their words: "warm", "apologetic", "urgent", or "neutral".

Examples (signs in order, grammar):
- "I need to sleep now." -> NOW I NEED SLEEP (statement)
- "Is your car new?" -> YOUR CAR NEW (yn)
- "My bus is late." -> MY BUS LATER (statement)
- "How did you learn to sign?" -> YOU LEARN SIGN HOW (wh)
- "We never close on Sunday." -> {{"kind": "spell", "value": "SUNDAY"}} WE CLOSE NEVER (neg)
- "My grandmother taught me how to cook." -> MY GRANDMOTHER TEACH I COOK (statement)
- "To be or not to be?" (a quote: sign its meaning) -> LIVE OR DIE WHICH (wh)
- "It's raining cats and dogs." (an idiom: sign its meaning) -> RAIN HEAVY (statement)
Reply as JSON: {{"tokens": [{{"kind": "sign" | "spell", "value": "..."}}], "grammar": "...", "tone": "..."}}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "tokens": {"type": "array", "items": {
            "type": "object",
            "properties": {"kind": {"type": "string", "enum": ["sign", "spell"]}, "value": {"type": "string"}},
            "required": ["kind", "value"], "additionalProperties": False}},
        "grammar": {"type": "string", "enum": ["yn", "wh", "neg", "statement"]},
        "tone": {"type": "string", "enum": ["warm", "apologetic", "urgent", "neutral"]},
    },
    "required": ["tokens", "grammar", "tone"],
    "additionalProperties": False,
}


class PlanError(Exception):
    pass


def _letters(s: str) -> str:
    return re.sub(r"[^A-Z]", "", s.upper())


def validate_plan(raw: dict, vocab: Vocab, english: str, lenient: bool) -> SignPlanMsg:
    """Strict: a non-vocabulary sign, or a sign repeated back to back when the speaker repeated no word ("to be or not
    to be" -> NOT NOT NOT, "grandmother" -> MOTHER MOTHER), raises PlanError. Lenient: the sign becomes a fingerspelled
    word and the repeat is dropped."""
    # shortcut: any doubled English word ("no, no") allows any doubled sign; compare per word if that ever misfires
    may_repeat = re.search(r"\b(\w+)\W+\1\b", english.lower()) is not None
    tokens: list[SignToken] = []
    for t in raw.get("tokens") or []:
        value = str(t.get("value", "")).strip()
        if t.get("kind") == "sign" and value in vocab.glosses:
            if tokens and tokens[-1].gloss == value and not may_repeat:
                if not lenient:
                    raise PlanError(f"repeated sign: {value!r}")
                continue
            tokens.append(SignToken(kind="sign", gloss=value))
            continue
        if t.get("kind") == "sign" and not lenient:
            raise PlanError(f"not in vocabulary: {value!r}")
        if word := _letters(value):
            tokens.append(SignToken(kind="spell", word=word))
    grammar = raw.get("grammar") if raw.get("grammar") in ("yn", "wh", "neg", "statement") else "statement"
    tone = raw.get("tone") if raw.get("tone") in ("warm", "apologetic", "urgent", "neutral") else "neutral"
    return SignPlanMsg(tokens=tokens, grammar=grammar, tone=tone, english=english)


class Gloss:
    def __init__(self, client, vocab: Vocab):
        self.client, self.vocab = client, vocab
        # Stable prefix: the vocabulary table never changes within a run, so it can be cached.
        # Offer only signs the avatar can show (curated, or with a motion clip); anything else gets fingerspelled.
        producible = [s for s in vocab.active if s.curated or (SIGNS_DIR / f"{s.gloss}.json").exists()]
        self.system = [{"type": "text", "text": SYSTEM.format(table=vocab.prompt_table(producible)),
                        "cache_control": {"type": "ephemeral"}}]

    async def plan(self, english: str) -> SignPlanMsg:
        # Two requests race (a strict-validation miss used to cost a second call in sequence: 2.4-2.8 s turns). The
        # schema-free one usually wins (a json_schema format cost 0.5-1.1 s); the schema one always returns JSON, for
        # when the model chats back instead ("I'm ready to translate...") on short input.
        tasks = [asyncio.create_task(self._ask(english, n, schema)) for n, schema in ((1, None), (2, SCHEMA))]
        near_miss = None  # parsed but used a non-vocabulary sign: kept for lenient validation
        try:
            for done in asyncio.as_completed(tasks):
                if (raw := await done) is None:
                    continue
                try:
                    return validate_plan(raw, self.vocab, english, lenient=False)
                except PlanError as e:
                    log.warning("gloss: %s", e)
                    near_miss = near_miss or raw
        finally:
            for t in tasks:
                t.cancel()
        if near_miss is not None:
            return validate_plan(near_miss, self.vocab, english, lenient=True)
        words = [w for w in (_letters(w) for w in english.split()) if w]
        return SignPlanMsg(tokens=[SignToken(kind="spell", word=w) for w in words], grammar="statement", tone="neutral", english=english)

    async def _ask(self, english: str, attempt: int, schema: dict | None) -> dict | None:
        try:
            # as data, not a bare message: a bare question invited the model to answer it ("why are you sad?" -> I NOT SAD)
            return await ask_json(self.client, self.system, json.dumps({"say": english}), schema)
        except LLMError as e:
            log.warning("gloss attempt %d failed: %s", attempt, e)
            return None
