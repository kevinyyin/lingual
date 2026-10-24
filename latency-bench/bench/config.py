"""Typed config loaded from config/bench.yaml (and secrets from latency-bench/.env only)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent  # latency-bench/
CONFIG_PATH = ROOT / "config" / "bench.yaml"
CORPUS_DIR = ROOT / "corpus"
RESULTS_DIR = ROOT / "results"


def load_env() -> None:
    """Load secrets from latency-bench/.env only (never a parent directory's .env)."""
    load_dotenv(ROOT / ".env", override=False)


class AudioCfg(BaseModel):
    chunk_ms: int = 20
    prepared_sample_rate: int = 24000
    pad_pre_s: float = 1.0
    pad_post_s: float = 6.0
    room_tone_dbfs: float = -60.0
    noise_snr_db: float = 10.0
    overlap_s: list[float] = [1.0, 2.0]
    output_gate_dbfs: float = -45.0


class VadCfg(BaseModel):
    threshold: float = 0.5
    min_silence_ms: int = 300
    speech_pad_ms: int = 0


class LLMCfg(BaseModel):
    provider: str = "openai"
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4.1-mini"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 400
    extra_body: dict[str, Any] = Field(default_factory=dict)


class EvsCfg(BaseModel):
    enabled: bool = True
    llm: str = "aligner"
    prompt: str = ""


class QualityCfg(BaseModel):
    chrf: bool = True
    llm_judge: bool = False
    judge_llm: str = "judge"


class AnalysisCfg(BaseModel):
    playback_policy: dict[str, Any] = Field(default_factory=lambda: {"mode": "none"})
    whisper_model: str = "small"
    whisper_compute_type: str = "int8"
    evs: EvsCfg = EvsCfg()
    quality: QualityCfg = QualityCfg()


class PrepareCfg(BaseModel):
    tts: str = "cartesia"
    cartesia_model: str = "sonic-3.6"
    openai_tts_model: str = "gpt-4o-mini-tts"
    openai_tts_voice: str = "coral"


class NetworkProbeCfg(BaseModel):
    samples: int = 5


class BenchConfig(BaseModel):
    seed: int = 1234
    concurrency: int = 1
    inter_run_delay_s: float = 1.0
    location: str = ""
    audio: AudioCfg = AudioCfg()
    vad: VadCfg = VadCfg()
    pairs: list[str] = ["es-en"]
    categories: list[str] = []
    network_probe: NetworkProbeCfg = NetworkProbeCfg()
    providers: dict[str, dict[str, Any]] = Field(default_factory=dict)
    llm: dict[str, LLMCfg] = Field(default_factory=dict)
    analysis: AnalysisCfg = AnalysisCfg()
    prepare: PrepareCfg = PrepareCfg()

    def provider_cfg(self, name: str) -> dict[str, Any]:
        return dict(self.providers.get(name, {}))

    def llm_cfg(self, role: str) -> LLMCfg:
        if role not in self.llm:
            raise KeyError(f"no llm role '{role}' in config")
        return self.llm[role]

    @property
    def resolved_location(self) -> str:
        return os.environ.get("BENCH_LOCATION") or self.location


def load_config(path: Path | None = None) -> BenchConfig:
    load_env()
    p = path or CONFIG_PATH
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return BenchConfig.model_validate(raw)


def config_raw_text(path: Path | None = None) -> str:
    return (path or CONFIG_PATH).read_text(encoding="utf-8")
