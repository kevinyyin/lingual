"""S4 extra condition: let the LLM put the emotion into the words. Each of the 12 café sentences is rewritten
by claude-haiku-5-5 for each emotion (same meaning, short, natural spoken English), then synthesized
(e) with the matching generation_config.emotion and (f) without any tag, on two voices. This separates the
effect of the words from the effect of the tag. Writes results/s4_rewrites.json; the clips are appended to
results/s4_tts_runs.jsonl by s4_tts.py (ids starting with 'e'/'f')."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import anthropic

sys.path.insert(0, os.path.dirname(__file__))
from s4_tts import EMOTIONS, SENTENCES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SYSTEM = ("Rewrite a café customer's line so the emotion is audible from the words alone, as natural spoken English. "
          "Keep the same request and facts, keep it under 16 words, no stage directions, no emoji. "
          "neutral = plain and polite; happy = warm and upbeat; sad = low, tired or disappointed; angry = curt and irritated.")
SCHEMA = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"], "additionalProperties": False}


def main():
    out_p = ROOT / "results/s4_rewrites.json"
    if out_p.exists():
        print("exists")
        return
    c = anthropic.Anthropic(max_retries=1)
    rw = {}
    for i, s in enumerate(SENTENCES):
        for e in EMOTIONS:
            r = c.messages.create(model="claude-haiku-5-5", max_tokens=200, system=SYSTEM,
                                  messages=[{"role": "user", "content": f"emotion: {e}\nline: {s}"}],
                                  output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
                                  thinking={"type": "disabled"})
            rw[f"{i}|{e}"] = json.loads(next(b.text for b in r.content if b.type == "text"))["text"]
            print(i, e, rw[f"{i}|{e}"], flush=True)
    json.dump(rw, open(out_p, "w"), indent=1)


if __name__ == "__main__":
    main()
