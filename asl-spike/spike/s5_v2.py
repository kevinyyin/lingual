"""S5 variant 2: same items as s5_llm.py, two changes that the first pass suggested.
G2E: the barista's previous utterance is given as context.
E2G: worked examples in the prompt, and one repair retry when validation finds glosses outside the
vocabulary (the invalid glosses are sent back; latency includes the retry).
Writes results/s5_g2e_v2.jsonl and results/s5_e2g_v2.jsonl."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import anthropic
import s5_llm as s

CTX = ["Hi there, welcome in! What can I get started for you?", "Sure thing. Hot or iced?",
       "And what size: small, medium or large?", "Do you want any milk in it?",
       "We do. We have oat, almond, soy, and regular.", "Got it. Anything to eat? The muffins are fresh this morning.",
       "Blueberry, chocolate chip, and banana nut.", "Of course. Is that for here or to go?",
       "Can I get a name for the order?", "Thanks, Maya. That comes to seven fifty.",
       "That's $7.50. How would you like to pay?", "Yep, just tap right here. Do you want a receipt?",
       "Your drink will be ready at the end of the counter.", "Here you go, enjoy!",
       "Hi, what can I get you?", "What can I get you?", "What would you like?", "Do you want milk with it?",
       "Anything else?", "Can I help you with anything?", "Anything else for you?", "Hi! What can I get you?",
       "Anything to eat?", "Your order will be up soon.", "Is everything okay?", "Is everything okay with your drink?",
       "How would you like to pay?", "For here or to go?", "What can I get you?", "What can I get started for you?"]
CTX += [CTX[0], CTX[3], CTX[4], CTX[6], CTX[6], CTX[8], CTX[9], "How would you like to pay?", CTX[11], CTX[13]]
assert len(CTX) == len(s.G2E_ITEMS)

E2G_EXAMPLES = """
Examples (input -> signs, nmm in brackets: r=raise, l=lower, n=neutral, h=headshake):
"Do you want it hot?" -> HOT[r] YOU[r] WANT[r]
"What size do you want?" -> SIZE[l] YOU[l] WANT[l] WHAT[l]
"That's twelve fifty." -> COST[n] TWELVE[n] DOLLAR[n] FS-FIFTY[n]   (12.50: number glosses only exist for 1-20; fingerspell or use digits)
"We're out of almond milk, sorry." -> ALMOND MILK[n] FINISH[h] NONE[h] SORRY[n]
"Is that everything?" -> FINISH[r] ALL[r]
"It's on the left." -> FS-LEFT[n]   (a word not in the vocabulary is fingerspelled, never invented)
Every output must contain at least one sign."""


def main():
    client = anthropic.Anthropic(max_retries=0)
    vocab_set = set(s.VOCAB)
    sys2 = s.E2G_SYSTEM.format(vocab=", ".join(s.VOCAB)) + E2G_EXAMPLES
    with open(s.RES / "s5_g2e_v2.jsonl", "w", encoding="utf-8") as f:
        for i, ((g, st, neg, aff, inten, ref, err), ctx) in enumerate(zip(s.G2E_ITEMS, CTX)):
            user = (f"barista just said: {ctx}\nglosses: {g}\nsentence_type: {st}\nnegation: {str(neg).lower()}\n"
                    f"affect: {aff} {inten}")
            rec = {"i": i, "glosses": g, "context": ctx, "reference": ref, "error": err}
            try:
                out, ms, usage = s.call(client, s.G2E_SYSTEM + "\nUse the barista's last utterance as context for short answers.", user, s.G2E_SCHEMA)
                rec.update(output=out, latency_ms=ms, usage=usage)
            except anthropic.APIStatusError as e:
                rec.update(failure=f"{e.status_code}: {str(e)[:200]}")
                if e.status_code in (401, 402, 403, 429):
                    f.write(json.dumps(rec) + "\n"); sys.exit(2)
            f.write(json.dumps(rec) + "\n")
            print("g2e", i, (rec.get("output") or {}).get("english"))
    with open(s.RES / "s5_e2g_v2.jsonl", "w", encoding="utf-8") as f:
        for i, eng in enumerate(s.E2G_ITEMS):
            rec = {"i": i, "english": eng}
            try:
                t0 = time.perf_counter_ns()
                out, ms, usage = s.call(client, sys2, eng, s.E2G_SCHEMA)
                bad = [x["gloss"] for x in (out or {}).get("signs", []) if x["gloss"] not in vocab_set and not s.FS_RE.match(x["gloss"])]
                rec["first_invalid"] = bad
                retried = False
                if bad or not (out or {}).get("signs"):
                    retried = True
                    fix = (f"{eng}\n\nYour previous answer used glosses that are not in the vocabulary: {bad or 'none (empty answer)'}. "
                           f"Previous answer: {json.dumps(out)}. Replace each with a vocabulary gloss or FS-<WORD>, keep the meaning.")
                    out, _, usage2 = s.call(client, sys2, fix, s.E2G_SCHEMA)
                    bad = [x["gloss"] for x in (out or {}).get("signs", []) if x["gloss"] not in vocab_set and not s.FS_RE.match(x["gloss"])]
                ms = (time.perf_counter_ns() - t0) / 1e6
                rec.update(output=out, latency_ms=ms, retried=retried, invalid_glosses=bad, rejected=bool(bad) or not (out or {}).get("signs"))
            except anthropic.APIStatusError as e:
                rec.update(failure=f"{e.status_code}: {str(e)[:200]}", rejected=True)
                if e.status_code in (401, 402, 403, 429):
                    f.write(json.dumps(rec) + "\n"); sys.exit(2)
            f.write(json.dumps(rec) + "\n")
            print("e2g", i, " ".join(x["gloss"] for x in (rec.get("output") or {}).get("signs", [])), rec.get("first_invalid"), rec.get("invalid_glosses"))


if __name__ == "__main__":
    main()
