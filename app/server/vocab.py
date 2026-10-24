"""Loads vocab/vocab.yaml (curated, committed) plus vocab/vocab_extended.yaml when present (every ASL Citizen and WLASL
sign, generated locally by training.build_vocab from data that may not be redistributed)."""

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT = Path(__file__).resolve().parent.parent / "vocab" / "vocab.yaml"
EXTENDED = DEFAULT.with_name("vocab_extended.yaml")


@dataclass(frozen=True)
class Sign:
    gloss: str
    english: str
    category: str
    asl_citizen: str
    active: bool = True
    avatar_ok: bool = True
    curated: bool = True  # False for vocab_extended.yaml entries (generated, not hand-checked)


class Vocab:
    def __init__(self, signs: list[Sign]):
        self.all = signs
        self.active = [s for s in signs if s.active]  # classifier label order
        self.glosses = {s.gloss for s in self.active}
        self._by_gloss = {s.gloss: s for s in signs}
        self._index = {s.gloss: i for i, s in enumerate(self.active)}

    def index(self, gloss: str) -> int:
        return self._index[gloss]

    def get(self, gloss: str) -> Sign | None:
        return self._by_gloss.get(gloss)

    def prompt_table(self, signs: list[Sign] | None = None) -> str:
        return "\n".join(f"{s.gloss} — {s.english}" for s in (self.active if signs is None else signs))


def load_vocab(path: Path | str = DEFAULT, extended: Path | str | None = "auto") -> Vocab:
    """extended="auto" merges vocab_extended.yaml only for the default curated file; curated entries win."""
    signs = [Sign(**s) for s in yaml.safe_load(Path(path).read_text(encoding="utf-8"))["signs"]]
    ext = EXTENDED if extended == "auto" and Path(path) == DEFAULT else None if extended == "auto" else extended
    if ext is not None and Path(ext).exists():
        have = {s.gloss for s in signs}
        signs += [Sign(**s) for s in yaml.safe_load(Path(ext).read_text(encoding="utf-8"))["signs"] if s["gloss"] not in have]
    return Vocab(signs)
