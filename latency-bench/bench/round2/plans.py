"""Round-2 experiment plans: which systems (conditions), clips, repetitions and concurrency.

Latency medians come only from concurrency-1 plans with 1 s between runs. Stall-count
plans (exp2a, exp2b) use concurrency 4 and are labelled as such in batch metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bench.corpus import PreparedClip, load_index
from bench.runner import System

SPEECH = ("short", "medium", "noisy", "long")
NEW = ("two_utterances", "hesitation", "three_turns")


@dataclass
class Plan:
    experiment: str
    systems: list[System]
    categories: tuple[str, ...]
    runs: int
    concurrency: int = 1
    pair: str = "es-en"
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def clips(self) -> list[PreparedClip]:
        idx = load_index()
        return sorted([c for c in idx.values() if c.pair == self.pair and c.category in self.categories],
                      key=lambda c: c.id)


def oa(label: str, **ov: Any) -> System:
    return System(f"openai/{label}", "openai", ov)


def casc(label: str, **ov: Any) -> System:
    return System(f"cascade/{label}", "cascade", {"translate_mode": "incremental", **ov})


SINGLE: dict[str, dict[str, Any]] = {
    "single-mbd0": {"tts_context": "per_turn", "tts": {"max_buffer_delay_ms": 0}},
    "single-mbd200": {"tts_context": "per_turn", "tts": {"max_buffer_delay_ms": 200}},
    # flush after each fragment instead of tuning the buffer: Cartesia's default 3000 ms buffer
    "single-flush": {"tts_context": "per_turn", "tts_flush_each": True, "tts": {"max_buffer_delay_ms": 3000}},
}


def exp2_trailing(chunk: int, kind: str) -> dict[str, Any]:
    return {"append_chunk_ms": chunk, "pad_post_s": 30, "trailing_audio": kind, "close_timeout_s": 40}


def plan(name: str, **kw: Any) -> Plan:
    if name == "exp2a":
        return Plan("exp2a", [oa("a20-rt60", **exp2_trailing(20, "room_tone_m60")),
                              oa("a200-rt60", **exp2_trailing(200, "room_tone_m60"))],
                    ("short", "medium"), runs=20, concurrency=4,
                    notes="stall study phase 2a; concurrency 4 (counts events, not latency medians)")
    if name == "exp2b":
        c = int(kw["chunk"])
        return Plan("exp2b", [oa(f"a{c}-rt60", **exp2_trailing(c, "room_tone_m60")),
                              oa(f"a{c}-zero", **exp2_trailing(c, "digital_silence")),
                              oa(f"a{c}-nb50", **exp2_trailing(c, "noise_bed_m50"))],
                    ("short", "medium"), runs=10, concurrency=4,
                    notes="stall study phase 2b; concurrency 4")
    if name == "exp2c":
        c, kind = int(kw["chunk"]), kw["trailing"]
        return Plan("exp2c", [oa(f"best-a{c}-{kind}", **exp2_trailing(c, kind))], SPEECH, runs=5,
                    notes="best exp2 condition at concurrency 1 for clean medians")
    if name == "exp3":
        c = int(kw.get("chunk", 20))
        base = {"append_chunk_ms": c}
        systems = [oa(f"flush-none-a{c}", flush_policy="none", **base)]
        systems += [oa(f"flush-pause{ms}-a{c}", flush_policy="on_pause", min_silence_ms=ms, **base)
                    for ms in (400, 600, 800)]
        systems += [oa(f"flush-watchdog1200-a{c}", flush_policy="watchdog", watchdog_ms=1200, min_silence_ms=400, **base)]
        return Plan("exp3", systems, ("medium", "noisy", "long") + NEW, runs=5,
                    notes="flush on pause / watchdog via session rotation; concurrency 1")
    if name == "exp5a":
        return Plan("exp5a", [casc("baseline"), casc("single-mbd0", **SINGLE["single-mbd0"]),
                              casc("single-mbd200", **SINGLE["single-mbd200"]),
                              casc("single-flush", **SINGLE["single-flush"])],
                    SPEECH + NEW, runs=5, notes="cascade TTS context variants; phrase-by-phrase mode; concurrency 1")
    if name == "exp5b":
        best = kw["best"]  # winning single-context variant from exp5a
        b = SINGLE[best]
        return Plan("exp5b", [casc(best, **b), casc(f"{best}+concise", **b, prompt_variant="concise"),
                              casc(f"{best}+concise+adaptive", **b, prompt_variant="concise",
                                   adaptive_speed={"enabled": True})],
                    SPEECH + NEW, runs=5, notes="concise prompt and adaptive speed on the best single context")
    if name == "exp6":
        base = dict(kw["base"])  # best exp5 configuration (overrides)
        systems = [casc("gpt41mini-serial", **base)]
        systems.append(casc("gpt41mini-pipelined", **base, pipeline_fragments=True))
        systems.append(casc("groq-serial", **base, llm="translate_groq"))
        systems.append(casc("groq-pipelined", **base, llm="translate_groq", pipeline_fragments=True))
        return Plan("exp6", systems, SPEECH, runs=5,
                    notes="translation host and pipelined fragments; phrase-by-phrase mode; concurrency 1")
    if name == "exp4":
        return Plan("exp4", [System("gemini", "gemini", {}), oa("none-control")],
                    SPEECH + NEW + ("same_language", "silence"), runs=5,
                    notes="Gemini Live Translate vs OpenAI control in one randomized batch; concurrency 1")
    raise KeyError(name)
