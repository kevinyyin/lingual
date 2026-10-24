"""Event models.

`ProviderEvent` is the in-memory event every provider yields while a run is live.
The `*Record` pydantic models are the on-disk schema of results/raw/<run_id>.jsonl:

    line 1      RunHeader        (what was run, on what machine/network, source timing)
    line 2      ConnectRecord    (session setup timing, kept apart from translation lag)
    line 3      SendRecordModel  (planned + actual send time of every source chunk)
    lines 4..   EventRecord      (every provider event, in arrival order)
    last line   RunFooter        (status: ok | failed | unsupported, error text)

Output audio bytes are written to results/audio/<run_id>.wav in arrival order; each
audio EventRecord stores its sample offset/length in that file, so the output stream can
be reconstructed exactly as it arrived.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

EventKind = Literal[
    "session_ready", "audio_out", "text_out_delta", "source_text_delta", "stage_mark", "error", "closed"
]


@dataclass(slots=True)
class ProviderEvent:
    kind: str
    t_recv_ns: int
    payload_size: int = 0
    pcm: bytes | None = None          # PCM16 LE mono, for audio_out
    sample_rate: int | None = None    # for audio_out
    text: str | None = None
    stage: str | None = None          # for stage_mark, e.g. "stt.end_of_turn"
    meta: dict[str, Any] = field(default_factory=dict)


class SourceTiming(BaseModel):
    """Speech onset/offset from silero-vad on the prepared (padded) source clip."""

    sample_rate: int
    n_samples: int
    onset_sample: int | None
    offset_sample: int | None
    segments: list[tuple[int, int]] = Field(default_factory=list)


class RunHeader(BaseModel):
    type: Literal["header"] = "header"
    run_id: str
    batch_id: str
    provider: str
    provider_config: dict[str, Any] = Field(default_factory=dict)
    clip_id: str
    pair: str
    category: str
    synthetic: bool
    repetition: int
    order_index: int
    source_lang: str
    target_lang: str
    chunk_ms: int
    input_sample_rate: int
    chunk_samples: int
    n_chunks: int
    source: SourceTiming           # in prepared-clip sample units (see sample_rate)
    machine: dict[str, Any] = Field(default_factory=dict)
    network: dict[str, Any] = Field(default_factory=dict)
    config_sha: str = ""
    clock: str = "time.perf_counter_ns"
    # round 2 (defaults reproduce round-1 behaviour, so round-1 files parse unchanged)
    condition: str | None = None          # named experimental condition, e.g. "openai/a200-silence"
    experiment: str | None = None         # e.g. "exp2a"
    send_alignment: str = "start"         # "start": chunk sent at its first sample; "end": when fully captured
    clip_transform: dict[str, Any] = Field(default_factory=dict)  # e.g. {"pad_post_s": 30, "trailing_audio": ...}


class ConnectRecord(BaseModel):
    type: Literal["connect"] = "connect"
    t_connect_start_ns: int
    t_session_ready_ns: int | None


class SendRecordModel(BaseModel):
    type: Literal["sends"] = "sends"
    planned_ns: list[int]
    sent_ns: list[int]
    max_lateness_ms: float
    drift_ms: float
    p99_lateness_ms: float = 0.0


class EventRecord(BaseModel):
    type: Literal["event"] = "event"
    kind: str
    t_recv_ns: int
    payload_size: int = 0
    sample_rate: int | None = None
    audio_offset: int | None = None   # sample offset into results/audio/<run_id>.wav
    n_samples: int | None = None
    text: str | None = None
    stage: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class RunFooter(BaseModel):
    type: Literal["footer"] = "footer"
    status: Literal["ok", "failed", "unsupported", "skipped"]
    error: str | None = None
    t_end_ns: int | None = None
    output_audio_file: str | None = None
    output_sample_rate: int | None = None


@dataclass
class RunData:
    header: RunHeader
    connect: ConnectRecord | None
    sends: SendRecordModel | None
    events: list[EventRecord]
    footer: RunFooter | None
    path: Path | None = None

    @property
    def ok(self) -> bool:
        return self.footer is not None and self.footer.status == "ok"


def write_jsonl(path: Path, records: list[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")


def read_run(path: Path) -> RunData:
    header = connect = sends = footer = None
    events: list[EventRecord] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            t = d.get("type")
            if t == "header":
                header = RunHeader.model_validate(d)
            elif t == "connect":
                connect = ConnectRecord.model_validate(d)
            elif t == "sends":
                sends = SendRecordModel.model_validate(d)
            elif t == "event":
                events.append(EventRecord.model_validate(d))
            elif t == "footer":
                footer = RunFooter.model_validate(d)
    if header is None:
        raise ValueError(f"{path}: missing header")
    return RunData(header, connect, sends, events, footer, path)
