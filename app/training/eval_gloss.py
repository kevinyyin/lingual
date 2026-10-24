"""English -> ASL sign plan, checked on hundreds of generated sentences against the live model (needs ANTHROPIC_API_KEY).

Each template knows the right answer's shape: the grammar (wh / yn / neg / statement), signs that must appear, and
ASL order rules (wh-sign last, time sign first). A plan fails a check when it misses one, adds YES/NO the speaker never
said, or falls back to fingerspelling everything (both model calls failed).

Run: uv run python -m training.eval_gloss [--per-template 12] [--concurrency 8]
Writes data/eval/gloss_results.jsonl and data/eval/gloss_report.md.
"""

import argparse
import asyncio
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import anthropic
import numpy as np
from dotenv import load_dotenv

from server.gloss import Gloss
from server.vocab import load_vocab

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "data" / "eval"
WH = {"WHAT", "WHERE", "WHEN", "WHO", "WHY", "HOW", "WHICH", "HOW-MANY"}

FOODS = ["COFFEE", "TEA", "WATER", "MILK", "JUICE", "BREAD", "CHEESE", "CAKE", "COOKIE", "SANDWICH", "PIZZA", "APPLE"]
PLACES = ["HOME", "SCHOOL", "STORE", "BATHROOM", "HOSPITAL", "RESTAURANT"]
FEELINGS = ["HAPPY", "SAD", "ANGRY", "TIRED", "SICK", "EXCITED", "SCARED"]
PEOPLE = ["FRIEND", "FAMILY", "MOTHER", "FATHER", "BABY", "DOCTOR"]
TIMES = ["TOMORROW", "YESTERDAY", "TODAY", "LATER", "NOW"]
NAMES = ["PRIYA", "OMAR", "JOSE", "MAYA", "KENJI", "ANA"]


def english(vocab, g: str) -> str:
    return vocab.get(g).english


# (template, slot lists, expected grammar, required sign slots, extra required signs, rule)
TEMPLATES = [
    ("Do you want {a}?", [FOODS], "yn", ["a"], ["WANT"], None),
    ("Would you like some {a}?", [FOODS], "yn", ["a"], [], None),
    ("Where is the {a}?", [PLACES], "wh", ["a"], ["WHERE"], "wh_last"),
    ("How do I get to the {a}?", [PLACES], "wh", ["a"], ["HOW"], "wh_last"),
    ("I don't like {a}.", [FOODS], "neg", ["a"], ["LIKE"], "has_neg"),
    ("We don't have any {a} today.", [FOODS], "neg", ["a"], ["HAVE"], "has_neg"),
    ("I need to go to the {a} {b}.", [PLACES, TIMES], "statement", ["a", "b"], ["NEED", "GO"], "time_first"),
    ("{b} I will buy {a}.", [FOODS, TIMES], "statement", ["a", "b"], ["BUY"], "time_first"),
    ("Why are you {a}?", [FEELINGS], "wh", ["a"], ["WHY"], "wh_last"),
    ("Is your {a} {b}?", [PEOPLE, FEELINGS], "yn", ["a", "b"], ["YOUR"], None),
    ("My {a} is {b}.", [PEOPLE, FEELINGS], "statement", ["a", "b"], ["MY"], None),
    ("Can you help my {a}?", [PEOPLE], "yn", ["a"], ["HELP"], None),
    ("Who is your {a}?", [PEOPLE], "wh", ["a"], ["WHO"], "wh_last"),
    ("My name is {n}.", [NAMES], "statement", [], ["NAME"], "spelled_name"),
    ("Nice to meet you, {n}!", [NAMES], "statement", [], ["NICE", "MEET"], "spelled_name"),
    ("I never eat {a} at night.", [FOODS], "neg", ["a"], ["NIGHT"], "has_neg"),
    ("When will your {a} come?", [PEOPLE], "wh", ["a"], ["WHEN", "COME"], "wh_last"),
    ("Thank you, the {a} was good.", [FOODS], "statement", ["a"], ["THANK-YOU", "GOOD"], None),
    ("Sorry, I don't understand.", [], "neg", [], ["SORRY", "UNDERSTAND"], "has_neg"),
    ("Which {a} is your favorite?", [FOODS], "wh", ["a"], ["FAVORITE"], "wh_last"),
    ("My grandmother taught me how to cook.", [], "statement", [], ["GRANDMOTHER", "TEACH", "COOK"], "no_collapse"),
    # sign the meaning: no word-for-word collapse into repeats or fingerspelling (a question either way is fine)
    ("To be or not to be?", [], ("wh", "yn"), [], [], "no_collapse"),
    ("Break a leg tonight!", [], "statement", [], [], "no_collapse"),
]


def cases(per_template: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for text, slots, grammar, req_slots, extra, rule in TEMPLATES:
        seen: set = set()
        want = per_template if slots else 1
        for _ in range(per_template * 5):
            if len(seen) >= want:
                break
            if slots and slots[0] is NAMES:
                fill = {"n": rng.choice(NAMES)}
            else:
                fill = {k: rng.choice(s) for k, s in zip("ab", slots)}
            key = tuple(sorted(fill.items()))
            if key in seen:
                continue
            seen.add(key)
            out.append({"template": text, "fill": fill, "grammar": grammar,
                        "required": [fill[k] for k in req_slots] + extra, "rule": rule})
    return out


def render(case: dict, vocab) -> str:
    f = {k: (v.title() if k == "n" else english(vocab, v).lower()) for k, v in case["fill"].items()}
    s = case["template"].format(**f)
    return s[0].upper() + s[1:]


def check(case: dict, plan, sentence: str) -> list[str]:
    signs = [t.gloss for t in plan.tokens if t.kind == "sign"]
    spelled = [t.word for t in plan.tokens if t.kind == "spell"]
    fails = []
    if not signs and len(spelled) >= len(sentence.split()) - 1:
        fails.append("fallback")
    allowed = case["grammar"] if isinstance(case["grammar"], (list, tuple)) else [case["grammar"]]
    if plan.grammar not in allowed:
        fails.append(f"grammar {plan.grammar}!={case['grammar']}")
    # ASL equivalents: NONE already means "don't have any"; "how do I get to X" is naturally X WHERE
    ok = {"HAVE": {"HAVE", "NONE"}, "HOW": {"HOW", "WHERE"}}
    missing = [g for g in case["required"] if not ok.get(g, {g}) & set(signs)]
    if missing:
        fails.append(f"missing {' '.join(missing)}")
    words = f" {sentence.lower()} "
    if "YES" in signs and " yes" not in words:
        fails.append("invented YES")
    if "NO" in signs and " no " not in words:
        fails.append("invented NO")
    if case["rule"] == "wh_last" and (not signs or (signs[-1] not in WH and signs[0] not in WH)):
        fails.append("wh-sign not first or last")  # both are grammatical ASL (and doubled)
    if case["rule"] == "has_neg" and not {"NOT", "NONE", "NEVER"} & set(signs):
        fails.append("no negation sign")
    if case["rule"] == "time_first":
        t = [i for i, g in enumerate(signs) if g in TIMES]
        if not t or t[0] != 0:
            fails.append("time sign not first")
    if case["rule"] == "no_collapse" and (len(set(signs)) < 2 or any(a == b for a, b in zip(signs, signs[1:]))
                                          or len(spelled) > len(signs)):
        fails.append("collapsed (repeats, under 2 signs, or mostly spelled)")
    if case["rule"] == "spelled_name" and case["fill"]["n"] not in spelled:
        fails.append("name not fingerspelled")
    return fails


async def run(per_template: int, concurrency: int) -> None:
    load_dotenv(ROOT / ".env")
    vocab = load_vocab()
    gloss = Gloss(anthropic.AsyncAnthropic(max_retries=0), vocab)
    sem = asyncio.Semaphore(concurrency)
    todo = cases(per_template)

    async def one(c):
        sentence = render(c, vocab)
        async with sem:
            t = time.perf_counter()
            plan = await gloss.plan(sentence)
            ms = (time.perf_counter() - t) * 1000
        return {**c, "sentence": sentence, "ms": ms, "fails": check(c, plan, sentence),
                "plan": " ".join(t.gloss or f"[{t.word}]" for t in plan.tokens), "plan_grammar": plan.grammar}

    rows = await asyncio.gather(*(one(c) for c in todo))
    EVAL.mkdir(parents=True, exist_ok=True)
    with open(EVAL / "gloss_results.jsonl", "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    by = defaultdict(list)
    for r in rows:
        by[r["template"]].append(r)
    lines = [f"# English -> ASL plan checks\n\n{len(rows)} sentences; "
             f"{np.mean([not r['fails'] for r in rows]):.1%} pass every check; "
             f"latency p50 {np.percentile([r['ms'] for r in rows], 50):.0f} ms, p90 {np.percentile([r['ms'] for r in rows], 90):.0f} ms\n",
             "| template | n | pass | most common failure |", "|---|---|---|---|"]
    for tpl, rs in by.items():
        fails = [f.split(" ")[0] + (" " + f.split(" ")[1] if f.startswith(("missing", "grammar")) else "") for r in rs for f in r["fails"]]
        common = max(set(fails), key=fails.count) if fails else ""
        lines.append(f"| {tpl} | {len(rs)} | {np.mean([not r['fails'] for r in rs]):.0%} | {common} |")
    lines += ["", "## Failures (sentence -> plan: why)", ""]
    lines += [f"- {r['sentence']} -> {r['plan']} ({r['plan_grammar']}): {'; '.join(r['fails'])}" for r in rows if r["fails"]]
    (EVAL / "gloss_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:len(by) + 3]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-template", type=int, default=12)
    ap.add_argument("--concurrency", type=int, default=8)
    a = ap.parse_args()
    asyncio.run(run(a.per_template, a.concurrency))


if __name__ == "__main__":
    main()
