"""Corpus manifest (corpus/manifest.yaml) and the prepared-clip index.

Base clips are listed under `clips:`; each has an audio file in corpus/audio/ (recorded
by a human, or synthesized by `bench prepare` and flagged `synthetic: true`).
Derived clips (noisy, overlap, silence) are listed under `derived:` and built by
`bench prepare` from base clips plus corpus/noise/.

Every clip belongs to exactly one `pair` (e.g. es-en): the session is configured as
pair.source -> pair.target, whatever language the clip actually contains. That is how
`same_language` (target-language speech into the session) and `overlap` (both
participants on one mic) are expressed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from bench.config import CORPUS_DIR
from bench.events import SourceTiming

MANIFEST_PATH = CORPUS_DIR / "manifest.yaml"
PREPARED_DIR = CORPUS_DIR / "prepared"
INDEX_PATH = PREPARED_DIR / "index.json"


class Clip(BaseModel):
    id: str
    pair: str
    category: str
    file: str | None = None            # relative to corpus/
    source_lang: str                   # language actually spoken ("es", "en", "es+en", "none")
    text: str | None = None            # source transcript (also used to synthesize)
    reference: str | None = None       # reference translation into the pair's target
    synthetic: bool = False
    voice: str | None = None           # TTS voice used when synthetic
    voice_group: str | None = None     # clips sharing a group are synthesized with the same voice
    notes: str | None = None

    @property
    def pair_src(self) -> str:
        return self.pair.split("-")[0]

    @property
    def pair_tgt(self) -> str:
        return self.pair.split("-")[1]


class DerivedSpec(BaseModel):
    id: str
    kind: str                          # noisy | overlap | silence | alias | concat | insert_pause
    pair: str
    category: str
    inputs: list[str] = Field(default_factory=list)   # base clip ids
    noise: str | None = None           # file in corpus/noise/ or "babble"
    duration_s: float | None = None    # for silence
    reference: str | None = None
    overlap_s: float | None = None
    gaps_s: list[float] = Field(default_factory=list)   # concat: room-tone gaps between inputs
    pause_ms: float | None = None      # insert_pause: length of the inserted pause
    after_word: int | None = None      # insert_pause: insert after this word index (0-based)


class Manifest(BaseModel):
    clips: list[Clip] = Field(default_factory=list)
    derived: list[DerivedSpec] = Field(default_factory=list)

    def by_id(self) -> dict[str, Clip]:
        return {c.id: c for c in self.clips}


def load_manifest(path: Path = MANIFEST_PATH) -> Manifest:
    with open(path, encoding="utf-8") as f:
        return Manifest.model_validate(yaml.safe_load(f) or {})


class PreparedClip(BaseModel):
    id: str
    pair: str
    category: str
    source_lang: str
    synthetic: bool
    file: str                          # relative to corpus/
    sample_rate: int
    duration_s: float
    timing: SourceTiming
    text: str | None = None
    reference: str | None = None
    sha256: str
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def path(self) -> Path:
        return CORPUS_DIR / self.file


def load_index(path: Path = INDEX_PATH) -> dict[str, PreparedClip]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: PreparedClip.model_validate(v) for k, v in data.items()}


def save_index(index: dict[str, PreparedClip], path: Path = INDEX_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({k: v.model_dump() for k, v in sorted(index.items())}, indent=1, ensure_ascii=False),
        encoding="utf-8",
    )
