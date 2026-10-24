"""Light quality sanity checks, to catch "fast but wrong". Never combined with latency."""

from __future__ import annotations

import hashlib
import json
import re

from sacrebleu.metrics import CHRF

from bench.analysis.align import CACHE
from bench.llm import LLMClient

_chrf = CHRF()


def chrf(hypothesis: str, reference: str) -> float | None:
    if not hypothesis or not reference:
        return None
    return round(_chrf.sentence_score(hypothesis, [reference]).score, 2)


async def judge_adequacy(llm: LLMClient, source: str, reference: str, hypothesis: str, tgt_lang: str) -> int | None:
    """Optional LLM judge: adequacy 1-5 of the hypothesis vs source/reference. Cached."""
    if not hypothesis:
        return None
    prompt = (f"Source utterance: {source}\nReference translation ({tgt_lang}): {reference}\n"
              f"Candidate translation (ASR of spoken output): {hypothesis}\n"
              "Rate the candidate's ADEQUACY (meaning preserved, ignore minor ASR noise and style) from 1 "
              '(unrelated) to 5 (complete and correct). Reply as JSON {"score": n}.')
    key = hashlib.sha256((prompt + llm.cfg.model).encode()).hexdigest()[:24]
    path = CACHE / "judge" / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())["score"]
    raw = await llm.complete("You are a strict translation quality judge. Reply with JSON only.", prompt)
    m = re.search(r'"score"\s*:\s*([1-5])', raw)
    score = int(m.group(1)) if m else None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"score": score}))
    return score
