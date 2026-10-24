"""Word timings (faster-whisper) and LLM word alignment, both cached on disk.

EVS is APPROXIMATE: it depends on ASR word boundaries on both sides and on an LLM's
judgement of which target word translates which source word.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from bench.audio import resample
from bench.config import RESULTS_DIR, BenchConfig
from bench.llm import LLMClient

CACHE = RESULTS_DIR / "cache"
_whisper = {}

WHISPER_LANGS = {"en", "es", "fr", "de", "ja", "zh", "it", "pt", "nl", "ru", "hi", "ko"}


def _sha(*parts: bytes | str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else p.encode("utf-8"))
    return h.hexdigest()[:24]


def whisper_model(cfg: BenchConfig):
    key = (cfg.analysis.whisper_model, cfg.analysis.whisper_compute_type)
    if key not in _whisper:
        from faster_whisper import WhisperModel

        _whisper[key] = WhisperModel(key[0], device="cpu", compute_type=key[1])
    return _whisper[key]


def transcribe_words(cfg: BenchConfig, x: np.ndarray, sr: int, language: str | None) -> dict:
    """{'text': str, 'words': [[word, start_s, end_s], ...]} with times relative to x[0]."""
    lang = language if language in WHISPER_LANGS else None
    x16 = resample(x.astype(np.float32), sr, 16000)
    key = _sha(x16.tobytes(), str(lang), cfg.analysis.whisper_model)
    path = CACHE / "asr" / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    segments, _info = whisper_model(cfg).transcribe(
        x16, language=lang, word_timestamps=True, beam_size=5, vad_filter=False,
        condition_on_previous_text=False,
    )
    words, texts = [], []
    for seg in segments:
        texts.append(seg.text)
        for w in seg.words or []:
            words.append([w.word.strip(), round(w.start, 3), round(w.end, 3)])
    res = {"text": "".join(texts).strip(), "words": words, "language": lang}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    return res


async def align_words(cfg: BenchConfig, llm: LLMClient, src_lang: str, tgt_lang: str,
                      src_words: list[str], tgt_words: list[str]) -> list[tuple[int, int]]:
    """LLM aligner: source content word index -> target word index. Cached by inputs."""
    if not src_words or not tgt_words:
        return []
    prompt = cfg.analysis.evs.prompt.format(
        src_lang=src_lang, tgt_lang=tgt_lang,
        src_words=" ".join(f"{i}:{w}" for i, w in enumerate(src_words)),
        tgt_words=" ".join(f"{i}:{w}" for i, w in enumerate(tgt_words)),
    )
    key = _sha(prompt, llm.cfg.model)
    path = CACHE / "align" / f"{key}.json"
    if path.exists():
        return [tuple(p) for p in json.loads(path.read_text(encoding="utf-8"))]
    raw = await llm.complete("You are a precise bilingual word aligner. Reply with JSON only.", prompt)
    m = re.search(r"\{.*\}", raw, re.S)
    pairs: list[tuple[int, int]] = []
    if m:
        try:
            for p in json.loads(m.group(0)).get("pairs", []):
                if isinstance(p, (list, tuple)) and len(p) == 2:
                    i, j = int(p[0]), int(p[1])
                    if 0 <= i < len(src_words) and 0 <= j < len(tgt_words):
                        pairs.append((i, j))
        except (ValueError, TypeError, json.JSONDecodeError):
            pairs = []
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pairs), encoding="utf-8")
    return pairs


def cached_json(name: str, key: str) -> Path:
    return CACHE / name / f"{key}.json"
