"""The test set's clip records and where everything lives (app/data/teaching, gitignored, local only)."""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "teaching"
CLIPS = DATA / "clips.jsonl"
FRAMES = DATA / "frames"
SOURCES = DATA / "sources.json"


@dataclass
class Clip:
    id: str  # "<source>:<source's own id>"
    source: str  # sources.json id (wlasl:<site>, aslsignbank, semlex, ...)
    gloss: str  # the app's gloss (vocab.yaml)
    path: str  # local video file
    signer: str = ""  # the source's signer id, when it has one
    url: str = ""
    licence: str = ""
    attribution: str = ""
    start_ms: float | None = None  # sub-clip of a longer video (None: whole file)
    end_ms: float | None = None
    extra: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return self.id.replace(":", "_").replace("/", "_")

    def frames_path(self, rate: str) -> Path:  # rate: "native" | "15fps"
        return FRAMES / rate / f"{self.key}.npz"


def load_clips(path: Path = CLIPS) -> list[Clip]:
    if not path.exists():
        return []
    return [Clip(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_clips(clips: list[Clip], path: Path = CLIPS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    seen, out = set(), []
    for c in clips:
        if c.id not in seen:
            seen.add(c.id)
            out.append(json.dumps(asdict(c)))
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def allowed_sources(path: Path = SOURCES) -> dict[str, dict]:
    """Sources the compliance review allowed (verdict "allow"), by id. Nothing is ingested from any other source."""
    if not path.exists():
        return {}
    return {s["id"]: s for s in json.loads(path.read_text(encoding="utf-8")) if s.get("verdict") == "allow"}
