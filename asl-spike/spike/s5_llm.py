"""S5: LLM glue in both directions with claude-haiku-5-5 (structured JSON output).

A  gloss -> English  (sign -> speech): 40 gloss sequences, 10 of them with one wrong or missing gloss.
B  English -> gloss with facial grammar (speech -> sign): 30 barista utterances; every output is
   validated against the vocabulary (plus FS-<WORD> fingerspelling).

Latency = wall time of one non-streaming request (perf_counter_ns), sequential, after one warm-up call.
Outputs results/s5_*.jsonl and results/s5_config.json.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import anthropic

sys.path.insert(0, os.path.dirname(__file__))
from vocab_src import V  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
MODEL = "claude-haiku-5-5"
VOCAB = sorted(V)

G2E_SYSTEM = """You turn recognized American Sign Language (ASL) glosses into the English sentence a Deaf customer
at a café counter means, so it can be spoken aloud to a hearing barista.
Input: the recognized gloss sequence (from an isolated-sign recognizer over a limited vocabulary; it can contain
a wrong or missing sign), the sentence type read from the signer's face (statement, yn = yes/no question,
wh = wh-question), a negation flag (headshake), and the signer's affect label and intensity (0-1).
Rules:
- Write natural, polite, first-person English, as the customer would say it. Keep it short.
- Use the sentence type for punctuation and word order: yn and wh questions end with "?".
- If negation is true, the sentence must be negative.
- FS-<WORD> is a fingerspelled word; use it as written (names in normal capitalization).
- ASL grammar differs from English (topic first, no articles, question sign at the end). Do not translate word by word.
- If a gloss does not fit the café context, it was probably misrecognized: infer the most plausible meaning and
  lower confidence. Never invent orders the glosses do not support.
- cartesia_emotion: one of neutral, calm, content, happy, sad, angry, scared. Start from the affect label, but
  use neutral when the affect contradicts the meaning of the words.
- confidence: 0-1, how sure you are the English matches what the signer meant."""

G2E_SCHEMA = {"type": "object", "properties": {
    "english": {"type": "string"},
    "cartesia_emotion": {"type": "string", "enum": ["neutral", "calm", "content", "happy", "sad", "angry", "scared"]},
    "confidence": {"type": "number"}},
    "required": ["english", "cartesia_emotion", "confidence"], "additionalProperties": False}

E2G_SYSTEM = """You convert a hearing barista's spoken English into American Sign Language (ASL) gloss for a signing
avatar, with facial grammar (non-manual markers).
Use ONLY glosses from this vocabulary, or FS-<WORD> to fingerspell a word that is not in it (names, brands,
menu items). Vocabulary: {vocab}
Rules:
- Use ASL word order (topic first, time first, question word at the end for wh-questions), no articles or "be" verbs.
- nmm.brows: "raise" on the signs of a yes/no question, "lower" on the signs of a wh-question, else "neutral".
- nmm.headshake: true on the negated part of a negative sentence.
- nmm.mouth: optional short mouth morpheme or null.
- span: an integer grouping consecutive signs that share one non-manual marker (0 = no marker).
- Keep it short; drop filler words. Numbers 1-20 have their own glosses; larger numbers and prices: use the
  number glosses that exist plus DOLLAR / CENT, or fingerspell."""

E2G_SCHEMA = {"type": "object", "properties": {"signs": {"type": "array", "items": {
    "type": "object", "properties": {
        "gloss": {"type": "string"},
        "nmm": {"type": "object", "properties": {
            "brows": {"type": "string", "enum": ["raise", "lower", "neutral"]},
            "headshake": {"type": "boolean"},
            "mouth": {"type": ["string", "null"]}},
            "required": ["brows", "headshake", "mouth"], "additionalProperties": False},
        "span": {"type": "integer"}},
    "required": ["gloss", "nmm", "span"], "additionalProperties": False}}},
    "required": ["signs"], "additionalProperties": False}

# (glosses, type, negation, affect, intensity, reference English, error note or None)
G2E_ITEMS = [
    ("HELLO IX-1P WANT COFFEE PLEASE", "statement", False, "happy", 0.5, "Hi, I'd like a coffee, please.", None),
    ("HOT", "statement", False, "neutral", 0.2, "Hot, please.", None),
    ("LARGE PLEASE", "statement", False, "neutral", 0.2, "Large, please.", None),
    ("YES YOU HAVE OAT MILK", "yn", False, "neutral", 0.3, "Yes. Do you have oat milk?", None),
    ("OAT MILK SUGAR NONE", "statement", True, "neutral", 0.3, "Oat milk, and no sugar.", None),
    ("MUFFIN WHAT KIND", "wh", False, "neutral", 0.3, "What kinds of muffins do you have?", None),
    ("BLUEBERRY ONE WARM CAN", "yn", False, "neutral", 0.3, "One blueberry. Can you warm it up?", None),
    ("HERE", "statement", False, "neutral", 0.2, "For here.", None),
    ("MY NAME FS-MAYA", "statement", False, "neutral", 0.2, "My name is Maya.", None),
    ("SORRY HOW-MUCH AGAIN PLEASE", "wh", False, "neutral", 0.4, "Sorry, how much? Again, please.", None),
    ("CARD PAY CAN", "yn", False, "neutral", 0.3, "Can I pay with a card?", None),
    ("RECEIPT NOT NEED THANK-YOU", "statement", True, "neutral", 0.3, "No receipt, thanks.", None),
    ("WHERE IX-1P UNDERSTAND NOT", "wh", True, "neutral", 0.5, "Where? I don't understand.", None),
    ("FINE THANK-YOU GOOD DAY", "statement", False, "happy", 0.7, "Okay, thank you! Have a good day.", None),
    ("TEA HOT SMALL PLEASE", "statement", False, "neutral", 0.3, "A small hot tea, please.", None),
    ("COFFEE ICE MEDIUM", "statement", False, "neutral", 0.3, "A medium iced coffee.", None),
    ("YOU HAVE DECAF", "yn", False, "neutral", 0.3, "Do you have decaf?", None),
    ("MILK NOT WANT", "statement", True, "neutral", 0.3, "I don't want milk.", None),
    ("COOKIE TWO PLEASE", "statement", False, "happy", 0.4, "Two cookies, please.", None),
    ("BATHROOM WHERE", "wh", False, "neutral", 0.4, "Where is the bathroom?", None),
    ("HOW-MUCH CAKE", "wh", False, "neutral", 0.3, "How much is the cake?", None),
    ("IX-1P DEAF WRITE CAN", "yn", False, "neutral", 0.3, "I'm Deaf. Can you write it down?", None),
    ("ALLERGY NUTS IX-1P MUFFIN HAVE NUTS", "yn", False, "scared", 0.5, "I'm allergic to nuts. Does the muffin have nuts?", None),
    ("WAIT HOW-MANY MINUTE", "wh", False, "neutral", 0.4, "How many minutes is the wait?", None),
    ("ORDER WRONG IX-1P WANT OAT MILK", "statement", False, "angry", 0.7, "My order is wrong, I wanted oat milk.", None),
    ("SORRY COFFEE COLD", "statement", False, "sad", 0.5, "Sorry, my coffee is cold.", None),
    ("CASH PAY CAN", "yn", False, "neutral", 0.3, "Can I pay with cash?", None),
    ("TABLE OUTSIDE HAVE", "yn", False, "neutral", 0.3, "Do you have tables outside?", None),
    ("SANDWICH WHAT KIND HAVE", "wh", False, "neutral", 0.3, "What kinds of sandwiches do you have?", None),
    ("LATTE TWO LARGE ALMOND MILK", "statement", False, "happy", 0.4, "Two large lattes with almond milk.", None),
    # simulated recognition errors: one wrong (confusable/off-topic) or missing gloss
    ("HELLO IX-1P WANT COFFEE CAR", "statement", False, "happy", 0.5, "Hi, I'd like a coffee, please.", "PLEASE -> CAR"),
    ("YES YOU HAVE MILK", "yn", False, "neutral", 0.3, "Yes. Do you have oat milk?", "OAT missing"),
    ("OAT MILK SUGAR", "statement", True, "neutral", 0.3, "Oat milk, and no sugar.", "NONE missing (headshake kept)"),
    ("MUFFIN WHAT FISH", "wh", False, "neutral", 0.3, "What kinds of muffins do you have?", "KIND -> FISH"),
    ("BLUEBERRY ONE WARM", "yn", False, "neutral", 0.3, "One blueberry. Can you warm it up?", "CAN missing"),
    ("MY NAME FS-MAIA", "statement", False, "neutral", 0.2, "My name is Maya.", "fingerspelling error Y->I"),
    ("SORRY HOW-MUCH AGAIN BEE", "wh", False, "neutral", 0.4, "Sorry, how much? Again, please.", "PLEASE -> BEE"),
    ("CARD PAY", "yn", False, "neutral", 0.3, "Can I pay with a card?", "CAN missing"),
    ("RECEIPT NEED THANK-YOU", "statement", True, "neutral", 0.3, "No receipt, thanks.", "NOT missing (headshake kept)"),
    ("FINE THANK-YOU GOOD DOG", "statement", False, "happy", 0.7, "Okay, thank you! Have a good day.", "DAY -> DOG"),
]

E2G_ITEMS = [
    "Hi there, welcome in! What can I get started for you?",
    "Sure thing. Hot or iced?",
    "And what size: small, medium or large?",
    "Do you want any milk in it?",
    "We do. We have oat, almond, soy, and regular.",
    "Got it. Anything to eat? The muffins are fresh this morning.",
    "Blueberry, chocolate chip, and banana nut.",
    "Of course. Is that for here or to go?",
    "Can I get a name for the order?",
    "Thanks, Maya. That comes to seven fifty.",
    "Seven dollars and fifty cents.",
    "Yep, just tap right here. Do you want a receipt?",
    "Your drink will be ready at the end of the counter.",
    "Right over there, by the window.",
    "Sorry, we're out of oat milk today.",
    "Would you like whipped cream on that?",
    "We don't take cash, only card.",
    "Is that everything?",
    "Where would you like to sit?",
    "It'll be about five minutes.",
    "Do you want it decaf?",
    "Who is the order for?",
    "Why don't you try the chocolate cake? It's really good.",
    "The bathroom is in the back, on the left.",
    "Do you need a straw?",
    "Sorry, I didn't understand. Can you show me again?",
    "How many sugars?",
    "We close at nine.",
    "That's not on the menu, sorry.",
    "Have a great day!",
]

FS_RE = re.compile(r"^FS-[A-Z]+$")


def call(client, system: str, user: str, schema: dict) -> tuple[dict | None, float, dict]:
    t0 = time.perf_counter_ns()
    kw = dict(model=MODEL, max_tokens=1024, system=system, messages=[{"role": "user", "content": user}],
              output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
              thinking={"type": "disabled"})
    r = client.messages.create(**kw)
    ms = (time.perf_counter_ns() - t0) / 1e6
    text = next((b.text for b in r.content if b.type == "text"), None)
    usage = {"input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens, "stop": r.stop_reason}
    return (json.loads(text) if text else None), ms, usage


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("skipped: no key")
        return
    RES.mkdir(exist_ok=True)
    client = anthropic.Anthropic(max_retries=0)
    vocab_set = set(VOCAB)
    e2g_system = E2G_SYSTEM.format(vocab=", ".join(VOCAB))
    json.dump({"model": MODEL, "effort": "low", "thinking": "disabled", "g2e_system": G2E_SYSTEM,
               "e2g_system": e2g_system, "g2e_schema": G2E_SCHEMA, "e2g_schema": E2G_SCHEMA,
               "vocab_size": len(VOCAB), "latency": "wall time of one non-streaming request, sequential, after warm-up"},
              open(RES / "s5_config.json", "w"), indent=1)
    call(client, G2E_SYSTEM, "glosses: HELLO\nsentence_type: statement\nnegation: false\naffect: neutral 0.1", G2E_SCHEMA)
    with open(RES / "s5_g2e.jsonl", "w", encoding="utf-8") as f:
        for i, (g, st, neg, aff, inten, ref, err) in enumerate(G2E_ITEMS):
            user = f"glosses: {g}\nsentence_type: {st}\nnegation: {str(neg).lower()}\naffect: {aff} {inten}"
            rec = {"i": i, "glosses": g, "sentence_type": st, "negation": neg, "affect": aff, "intensity": inten,
                   "reference": ref, "error": err}
            try:
                out, ms, usage = call(client, G2E_SYSTEM, user, G2E_SCHEMA)
                rec.update(output=out, latency_ms=ms, usage=usage)
            except anthropic.APIStatusError as e:
                rec.update(failure=f"{e.status_code}: {str(e)[:200]}")
                if e.status_code in (401, 402, 403, 429):
                    f.write(json.dumps(rec) + "\n")
                    print("API FAILURE", e.status_code)
                    sys.exit(2)
            except Exception as e:
                rec.update(failure=f"{type(e).__name__}: {e}"[:300])
            f.write(json.dumps(rec) + "\n")
            print("g2e", i, rec.get("output", rec.get("failure")))
    with open(RES / "s5_e2g.jsonl", "w", encoding="utf-8") as f:
        for i, eng in enumerate(E2G_ITEMS):
            rec = {"i": i, "english": eng}
            try:
                out, ms, usage = call(client, e2g_system, eng, E2G_SCHEMA)
                bad = [s["gloss"] for s in (out or {}).get("signs", [])
                       if s["gloss"] not in vocab_set and not FS_RE.match(s["gloss"])]
                rec.update(output=out, latency_ms=ms, usage=usage, invalid_glosses=bad, rejected=bool(bad) or not out)
            except anthropic.APIStatusError as e:
                rec.update(failure=f"{e.status_code}: {str(e)[:200]}", rejected=True)
                if e.status_code in (401, 402, 403, 429):
                    f.write(json.dumps(rec) + "\n")
                    print("API FAILURE", e.status_code)
                    sys.exit(2)
            except Exception as e:
                rec.update(failure=f"{type(e).__name__}: {e}"[:300], rejected=True)
            f.write(json.dumps(rec) + "\n")
            print("e2g", i, " ".join(s["gloss"] for s in (rec.get("output") or {}).get("signs", [])), rec.get("invalid_glosses"))


if __name__ == "__main__":
    main()
