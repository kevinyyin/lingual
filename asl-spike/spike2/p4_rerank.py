"""P4 (optional): Claude reranker for phrase mode (claude-haiku-5-5, structured JSON, low effort, thinking off).

Input per utterance: the barista's previous line (from the scenario; the line before each customer turn), the
top-5 free beam decodes (glosses), and the top-3 script candidates (glosses + their English). Output: the top-3
English sentences, each tagged with the script candidate id it came from (or "free"). Scored on 100 seeded
test script renderings: top-1 = the first returned sentence comes from the correct script line.
Usage: python p4_rerank.py <model_tag>
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE2, R2, ROOT, model_classes  # noqa: E402

SCHEMA = {"type": "object", "properties": {"ranked": {"type": "array", "items": {"type": "object", "properties": {
    "english": {"type": "string"}, "source": {"type": "string"}}, "required": ["english", "source"], "additionalProperties": False}}},
    "required": ["ranked"], "additionalProperties": False}
SYSTEM = ("A Deaf customer is signing to a hearing barista in a café. A sign recognizer produced candidate gloss sequences. "
          "Choose what the customer most likely said. Prefer a script candidate when it fits the recognized signs and the barista's "
          "previous line; use a free decode only if no script candidate fits. Return the top 3 as short natural English sentences, "
          "each with source = the script candidate id (e.g. \"S2\") or \"free\".")


def main(tag):
    import anthropic
    import torch
    import p4_decode as D
    from p5_eval import load, logprobs, script_lm_and_lines
    sys.path.insert(0, str(ROOT / "spike"))
    from s5_llm import G2E_ITEMS
    from s5_v2 import CTX
    if not os.environ.get("ANTHROPIC_API_KEY"):
        json.dump({"skipped": "no key"}, open(R2 / "p4_rerank.json", "w"))
        print("skipped: no key")
        return
    classes = model_classes()
    net = load(tag)
    lm, lines = script_lm_and_lines(classes)
    test_s = pickle.load(open(CACHE2 / "p2_test_script.pkl", "rb"))
    rng = np.random.default_rng(0)
    pick = rng.choice(len(test_s), 100, replace=False)
    client = anthropic.Anthropic(max_retries=1)
    rows = []
    for i in pick:
        it = test_s[i]
        lp = logprobs(net, it["X"])
        beams = D.beam_search(lp.numpy(), lm)[:5]
        free = [" ".join(classes[t] if t < 143 else "<unk>" for t in b[0]) for b in beams]
        order, ll, _ = D.script_decode(lp, lines, None, None)
        cands = [{"id": f"S{j}", "glosses": " ".join(classes[t] if t < 143 else "<unk>" for t in lines[j]), "english": G2E_ITEMS[j][5]} for j in order]
        user = json.dumps({"barista_previous_line": CTX[it["line"]], "free_decodes": free, "script_candidates": cands})
        t0 = time.perf_counter_ns()
        try:
            r = client.messages.create(model="claude-haiku-5-5", max_tokens=400, system=SYSTEM, messages=[{"role": "user", "content": user}],
                                       output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
                                       thinking={"type": "disabled"})
            ms = (time.perf_counter_ns() - t0) / 1e6
            out = json.loads(next(b.text for b in r.content if b.type == "text"))["ranked"]
            src = [o["source"] for o in out]
            rows.append({"line": it["line"], "signer": it["signer"], "script_top1_order": int(order[0]), "rerank_sources": src,
                         "rerank_top1": src[:1] == [f"S{it['line']}"], "rerank_top3": f"S{it['line']}" in src[:3],
                         "script_top1": int(order[0]) == it["line"], "latency_ms": ms,
                         "tokens": [r.usage.input_tokens, r.usage.output_tokens]})
        except Exception as e:
            rows.append({"line": it["line"], "error": f"{type(e).__name__}: {e}"[:200]})
            if any(w in str(e) for w in ("401", "402", "403", "429", "credit")):
                print("API FAILURE, stopping:", e)
                break
    ok = [r for r in rows if "error" not in r]
    lat = [r["latency_ms"] for r in ok]
    res = {"model": tag, "n": len(ok), "failures": len(rows) - len(ok),
           "script_top1_without_reranker": float(np.mean([r["script_top1"] for r in ok])),
           "rerank_top1": float(np.mean([r["rerank_top1"] for r in ok])), "rerank_top3": float(np.mean([r["rerank_top3"] for r in ok])),
           "latency_ms": {"median": float(np.median(lat)), "p90": float(np.percentile(lat, 90)), "max": float(np.max(lat))},
           "tokens_in_out": [int(sum(r["tokens"][0] for r in ok)), int(sum(r["tokens"][1] for r in ok))],
           "est_cost_usd": round(sum(r["tokens"][0] for r in ok) / 1e6 * 0.10 + sum(r["tokens"][1] for r in ok) / 1e6 * 0.50, 4),
           "rows": rows}
    json.dump(res, open(R2 / "p4_rerank.json", "w"), indent=1)
    print({k: v for k, v in res.items() if k != "rows"})


if __name__ == "__main__":
    main(sys.argv[1])
