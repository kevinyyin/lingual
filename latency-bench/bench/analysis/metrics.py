"""Every metric is computed here from raw recorded events only.

Time origin: all timestamps are perf_counter_ns() values from one run. Source events
(speech onset/offset, source words) are mapped onto that clock through the recorded
send time of the chunk that carried the sample.

Definitions
-----------
t_onset / t_offset   send time of the chunk containing the VAD speech onset / offset
                     sample.
start_lag            arrival of the first *audible* output sample (peak above
                     output_gate_dbfs) minus t_onset.
first_byte_lag       arrival of the first output audio event of any content minus t_onset.
end_lag_arrival      arrival of the last audible output sample minus t_offset.
end_lag_playback     a simulated listener plays output chunks back-to-back at 1x, each
                     starting at max(its arrival, end of the previous chunk). The moment
                     the last audible sample finishes playing, minus t_offset.
setup                t_session_ready - t_connect_start (reported separately).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from bench.audio import dbfs_to_amp
from bench.events import RunData

NS_PER_MS = 1_000_000


# ---------------------------------------------------------------- source mapping
def chunk_of_sample(sample: int, sample_rate: int, chunk_ms: int) -> int:
    return int(sample * 1000 // (sample_rate * chunk_ms))


def send_time_of_sample(sent_ns: list[int], sample: int, sample_rate: int, chunk_ms: int) -> int:
    """Send timestamp of the chunk that carried `sample` (spec: onset/offset mapping)."""
    k = min(max(chunk_of_sample(sample, sample_rate, chunk_ms), 0), len(sent_ns) - 1)
    return sent_ns[k]


def precise_time_of_sample(sent_ns: list[int], sample: int, sample_rate: int, chunk_ms: int) -> int:
    """Send time of the carrying chunk plus the sample's position inside it (used for
    word-level timing, where sub-chunk resolution matters)."""
    k = min(max(chunk_of_sample(sample, sample_rate, chunk_ms), 0), len(sent_ns) - 1)
    within = sample - k * sample_rate * chunk_ms / 1000
    return int(sent_ns[k] + within / sample_rate * 1e9)


# ---------------------------------------------------------------- output audio
@dataclass
class OutChunk:
    t_recv_ns: int
    n: int
    sr: int
    offset: int                 # sample offset in the run's output wav
    first_audible: int | None   # index within the chunk of first sample above gate
    last_audible: int | None
    seg: Any = None             # cascade segment id (if any)

    @property
    def dur_ns(self) -> int:
        return int(round(self.n * 1e9 / self.sr))


def audible_bounds(samples: np.ndarray, gate_dbfs: float) -> tuple[int | None, int | None]:
    thr = dbfs_to_amp(gate_dbfs) * 32768.0
    idx = np.flatnonzero(np.abs(samples.astype(np.int32)) > thr)
    if idx.size == 0:
        return None, None
    return int(idx[0]), int(idx[-1])


def output_chunks(run: RunData, pcm: np.ndarray | None, gate_dbfs: float) -> list[OutChunk]:
    """Audio events in arrival order, with audible-sample bounds from the captured wav."""
    out: list[OutChunk] = []
    for ev in run.events:
        if ev.kind != "audio_out" or not ev.n_samples:
            continue
        fa = la = None
        if pcm is not None and ev.audio_offset is not None:
            fa, la = audible_bounds(pcm[ev.audio_offset : ev.audio_offset + ev.n_samples], gate_dbfs)
        out.append(OutChunk(ev.t_recv_ns, ev.n_samples, ev.sample_rate or 24000, ev.audio_offset or 0,
                            fa, la, ev.meta.get("seg")))
    return out


def simulate_playback(chunks: list[OutChunk]) -> list[tuple[int, int]]:
    """(play_start_ns, play_end_ns) per chunk: back-to-back at 1x, never before arrival."""
    spans: list[tuple[int, int]] = []
    prev_end = -(2**62)
    for c in chunks:
        start = max(c.t_recv_ns, prev_end)
        end = start + c.dur_ns
        spans.append((start, end))
        prev_end = end
    return spans


def playback_time_of_output_sample(chunks: list[OutChunk], spans: list[tuple[int, int]], sample: int) -> int | None:
    """Simulated playback time of an output-wav sample index."""
    for c, (s, _e) in zip(chunks, spans):
        if c.offset <= sample < c.offset + c.n:
            return int(s + (sample - c.offset) * 1e9 / c.sr)
    if chunks and sample >= chunks[-1].offset + chunks[-1].n:
        c, (s, _e) = chunks[-1], spans[-1]
        return int(s + (sample - c.offset) * 1e9 / c.sr)
    return None


# ---------------------------------------------------------------- playback policies (round 2, exp 1)
BLOCK_MS = 20  # policy decisions are made per <= 20 ms block of output content


@dataclass
class PlaybackPolicy:
    """none: 1x. catchup: when queued unplayed audio exceeds threshold_ms, play at `speed`
    (capped at max_speed) until it drops below threshold_ms / 2. catchup_end: same, but
    speed-up is only allowed inside `speed_windows` (after a source utterance offset)."""

    mode: str = "none"
    threshold_ms: float = 500.0
    speed: float = 1.25
    max_speed: float = 1.5

    @property
    def label(self) -> str:
        if self.mode == "none":
            return "none"
        return f"{self.mode}@{self.speed:g}x/{self.threshold_ms:g}ms"

    @classmethod
    def from_cfg(cls, d: dict[str, Any] | None) -> "PlaybackPolicy":
        d = d or {}
        return cls(mode=d.get("mode", "none"), threshold_ms=float(d.get("threshold_ms", 500)),
                   speed=float(d.get("speed", 1.25)), max_speed=float(d.get("max_speed", 1.5)))


@dataclass
class PlayBlock:
    chunk: int        # index into the chunk list
    offset: int       # sample offset in the output wav
    n: int
    sr: int
    arrival_ns: int
    start_ns: int = 0
    end_ns: int = 0
    speed: float = 1.0


def simulate_playback_policy(chunks: list[OutChunk], policy: PlaybackPolicy | None = None,
                             speed_windows: list[tuple[int, int]] | None = None) -> list[PlayBlock]:
    """Play output content block by block, never before arrival, back to back.

    Backlog = content duration (at 1x) of audio that has arrived but not yet played,
    including the block about to start. Hysteresis: speed up when backlog > threshold,
    return to 1x when backlog < threshold / 2. A block of content duration d occupies
    d / speed of play-out time."""
    policy = policy or PlaybackPolicy()
    blocks: list[PlayBlock] = []
    for ci, c in enumerate(chunks):
        step = max(1, c.sr * BLOCK_MS // 1000)
        for o in range(0, c.n, step):
            blocks.append(PlayBlock(ci, c.offset + o, min(step, c.n - o), c.sr, c.t_recv_ns))
    content = [b.n * 1e9 / b.sr for b in blocks]
    prefix = np.concatenate([[0.0], np.cumsum(content)]) if blocks else np.zeros(1)
    fast, prev_end, j = False, -(2**62), 0
    thr = policy.threshold_ms * 1e6
    s_fast = min(policy.speed, policy.max_speed)
    for i, b in enumerate(blocks):
        t = max(b.arrival_ns, prev_end)
        while j < len(blocks) and blocks[j].arrival_ns <= t:
            j += 1
        queued = prefix[j] - prefix[i]
        if policy.mode != "none":
            if not fast and queued > thr:
                fast = True
            elif fast and queued < thr / 2:
                fast = False
        allowed = policy.mode == "catchup" or (
            policy.mode == "catchup_end" and any(a <= t < z for a, z in (speed_windows or [])))
        b.speed = s_fast if (fast and allowed) else 1.0
        b.start_ns = int(t)
        b.end_ns = int(round(t + content[i] / b.speed))
        prev_end = b.end_ns
    return blocks


def simulate_playback_tracks(chunks: list[OutChunk], track_of: list[Any], policy: PlaybackPolicy | None = None,
                             speed_windows: list[tuple[int, int]] | None = None) -> tuple[list[PlayBlock], float]:
    """Several parallel output streams (e.g. OpenAI sessions during rotation), each played
    on its own track at the policy's speed and mixed for the listener.

    Concatenating parallel real-time streams in arrival order would double the content per
    second of wall time (each session streams silent filler frames continuously), so every
    track is simulated separately. Returns all blocks (chunk indices refer to `chunks`) and
    the seconds during which two or more tracks were audible at once (overlap)."""
    blocks: list[PlayBlock] = []
    audible_spans: dict[Any, list[tuple[int, int]]] = {}
    for tid in dict.fromkeys(track_of):
        idx = [i for i, t in enumerate(track_of) if t == tid]
        sub = [chunks[i] for i in idx]
        for b in simulate_playback_policy(sub, policy, speed_windows):
            b.chunk = idx[b.chunk]
            blocks.append(b)
            c = chunks[b.chunk]
            if c.first_audible is not None:
                audible_spans.setdefault(tid, []).append((b.start_ns, b.end_ns))
    # overlap: time covered by >= 2 tracks' audible blocks
    edges = sorted([(a, 1) for spans in audible_spans.values() for a, _ in spans] +
                   [(z, -1) for spans in audible_spans.values() for _, z in spans])
    depth, last, overlap = 0, None, 0
    for t, d in edges:
        if depth >= 2 and last is not None:
            overlap += t - last
        depth += d
        last = t
    blocks.sort(key=lambda b: b.offset)
    return blocks, round(overlap / 1e9, 3)


def block_time_of_sample(blocks: list[PlayBlock], sample: int, end: bool = False) -> int | None:
    """Play time of an output-wav sample (its start, or its end if end=True)."""
    for b in blocks:
        if b.offset <= sample < b.offset + b.n:
            k = sample - b.offset + (1 if end else 0)
            return int(b.start_ns + k * 1e9 / b.sr / b.speed)
    return None


def policy_end_lag_ms(chunks: list[OutChunk], blocks: list[PlayBlock], t_off: int | None) -> float | None:
    aud = [c for c in chunks if c.last_audible is not None]
    if not aud or t_off is None:
        return None
    last = aud[-1]
    t = block_time_of_sample(blocks, last.offset + last.last_audible, end=True)
    return None if t is None else round((t - t_off) / NS_PER_MS, 3)


# ---------------------------------------------------------------- run metrics
def _ms(ns: float | int | None) -> float | None:
    return None if ns is None else round(ns / NS_PER_MS, 3)


def source_time_of_sample(run: RunData, sample: int) -> int:
    """Clock time of a prepared-clip sample on this run's send timeline.

    send_alignment 'start' (round 1, 20 ms chunks): the send time of the chunk carrying it.
    send_alignment 'end' (chunk sent when fully captured, e.g. 200 ms appends): the chunk's
    send time minus its duration plus the sample's position inside it, i.e. the moment a
    live microphone would have captured that sample."""
    h, s = run.header, run.sends
    st = h.source
    if getattr(h, "send_alignment", "start") == "end":
        k = min(max(chunk_of_sample(sample, st.sample_rate, h.chunk_ms), 0), len(s.sent_ns) - 1)
        within = sample - k * st.sample_rate * h.chunk_ms / 1000
        return int(s.sent_ns[k] - h.chunk_ms * NS_PER_MS + within / st.sample_rate * 1e9)
    return send_time_of_sample(s.sent_ns, sample, st.sample_rate, h.chunk_ms)


def source_times(run: RunData) -> tuple[int | None, int | None]:
    h, s = run.header, run.sends
    if s is None or not s.sent_ns:
        return None, None
    st = h.source
    on = source_time_of_sample(run, st.onset_sample) if st.onset_sample is not None else None
    off = source_time_of_sample(run, st.offset_sample) if st.offset_sample is not None else None
    return on, off


def utterance_times(run: RunData) -> list[tuple[int, int]]:
    """(onset, offset) clock times of each VAD speech segment (min silence 300 ms)."""
    if run.sends is None or not run.sends.sent_ns:
        return []
    return [(source_time_of_sample(run, a), source_time_of_sample(run, b)) for a, b in run.header.source.segments]


def utterance_metrics(run: RunData, chunks: list[OutChunk], blocks: list[PlayBlock]) -> list[dict[str, Any]]:
    """Per-utterance lags. Each audible output chunk is assigned to the utterance whose
    offset most recently precedes its arrival (chunks before the first offset go to
    utterance 0). End lag at playback = play-out end of the last audible sample assigned
    to the utterance minus that utterance's offset."""
    utts = utterance_times(run)
    out = []
    aud = [c for c in chunks if c.first_audible is not None]
    assigned: dict[int, list[OutChunk]] = {i: [] for i in range(len(utts))}
    for c in aud:
        u = 0
        for i, (_on, off) in enumerate(utts):
            if off <= c.t_recv_ns:
                u = i
        if utts:
            assigned[u].append(c)
    for i, (on, off) in enumerate(utts):
        cs = assigned[i]
        first_after = next((c for c in aud if c.t_recv_ns >= on), None)
        end = None
        if cs:
            t = block_time_of_sample(blocks, cs[-1].offset + cs[-1].last_audible, end=True)
            end = _ms(t - off) if t is not None else None
        out.append({"utt": i, "speech_s": round((off - on) / 1e9, 3), "end_lag_playback_ms": end,
                    "end_lag_arrival_ms": _ms(cs[-1].t_recv_ns - off) if cs else None,
                    "start_lag_ms": _ms(first_after.t_recv_ns - on) if first_after else None,
                    "n_chunks": len(cs)})
    return out


def latency_metrics(run: RunData, chunks: list[OutChunk]) -> dict[str, Any]:
    m: dict[str, Any] = {}
    c = run.connect
    m["setup_ms"] = _ms(c.t_session_ready_ns - c.t_connect_start_ns) if c and c.t_session_ready_ns else None
    t_on, t_off = source_times(run)
    m["source_speech_s"] = (
        round((run.header.source.offset_sample - run.header.source.onset_sample) / run.header.source.sample_rate, 3)
        if run.header.source.onset_sample is not None else 0.0
    )
    m["n_audio_events"] = len(chunks)
    m["output_audio_s"] = round(sum(ch.n / ch.sr for ch in chunks), 3)
    audible = [i for i, ch in enumerate(chunks) if ch.first_audible is not None]
    m["produced_output"] = bool(audible)
    m["audible_output_s"] = round(sum(
        (chunks[i].last_audible - chunks[i].first_audible + 1) / chunks[i].sr for i in audible), 3)

    spans = simulate_playback(chunks)
    first_any = chunks[0].t_recv_ns if chunks else None
    m["first_byte_lag_ms"] = _ms(first_any - t_on) if (first_any is not None and t_on is not None) else None
    if audible and t_on is not None:
        m["start_lag_ms"] = _ms(chunks[audible[0]].t_recv_ns - t_on)
        fa = chunks[audible[0]]
        m["start_lag_playback_ms"] = _ms(spans[audible[0]][0] + fa.first_audible * 1e9 / fa.sr - t_on)
    else:
        m["start_lag_ms"] = m["start_lag_playback_ms"] = None
    if audible and t_off is not None:
        last = chunks[audible[-1]]
        m["end_lag_arrival_ms"] = _ms(last.t_recv_ns - t_off)
        play_end = spans[audible[-1]][0] + (last.last_audible + 1) * 1e9 / last.sr
        m["end_lag_playback_ms"] = _ms(play_end - t_off)
    else:
        m["end_lag_arrival_ms"] = m["end_lag_playback_ms"] = None

    if run.sends:
        m["send_max_lateness_ms"] = run.sends.max_lateness_ms
        m["send_drift_ms"] = run.sends.drift_ms
    return m


def provider_texts(run: RunData) -> dict[str, str]:
    out = "".join(e.text or "" for e in run.events if e.kind == "text_out_delta")
    src = "".join(e.text or "" for e in run.events if e.kind == "source_text_delta")
    return {"provider_output_text": out.strip(), "provider_source_text": src.strip()}


# ---------------------------------------------------------------- cascade stages
STAGE_COMPONENTS = ["stt_finalize_ms", "translate_wait_ms", "tts_first_byte_ms", "tts_stream_rest_ms"]


def cascade_stages(run: RunData, chunks: list[OutChunk]) -> dict[str, Any]:
    """Stage timings for the cascade's final segment (the one carrying the last audible
    output), plus a decomposition of end_lag_arrival into sequential components."""
    marks = [e for e in run.events if e.kind == "stage_mark" and e.stage]
    if not any(e.stage.startswith(("stt.", "llm.", "tts.")) for e in marks):
        return {}
    audible = [c for c in chunks if c.first_audible is not None]
    seg = audible[-1].seg if audible else None
    if seg is None:
        segs = [e.meta.get("seg") for e in marks if e.stage == "llm.request"]
        seg = segs[-1] if segs else None

    def t(stage: str, **match: Any) -> int | None:
        for e in marks:
            if e.stage == stage and all(e.meta.get(k) == v for k, v in match.items()):
                return e.t_recv_ns
        return None

    def t_last(stage: str, **match: Any) -> int | None:
        hits = [e.t_recv_ns for e in marks if e.stage == stage and all(e.meta.get(k) == v for k, v in match.items())]
        return hits[-1] if hits else None

    req = next((e for e in marks if e.stage == "llm.request" and e.meta.get("seg") == seg), None)
    turn = req.meta.get("turn") if req else None
    t_req = req.t_recv_ns if req else None
    t_tok, t_llm_done = t("llm.first_token", seg=seg), t("llm.done", seg=seg)
    t_txt, t_aud, t_tts_done = t("tts.first_text", seg=seg), t("tts.first_audio", seg=seg), t("tts.done", seg=seg)
    t_eot = t_last("stt.end_of_turn", turn=turn) if turn is not None else t_last("stt.end_of_turn")
    _, t_off = source_times(run)
    t_last_audio = audible[-1].t_recv_ns if audible else None

    d: dict[str, Any] = {
        "cascade_final_seg": seg,
        "llm_first_token_ms": _ms(t_tok - t_req) if t_tok and t_req else None,
        "llm_complete_ms": _ms(t_llm_done - t_req) if t_llm_done and t_req else None,
        "tts_first_byte_raw_ms": _ms(t_aud - t_txt) if t_aud and t_txt else None,
        "tts_complete_ms": _ms(t_tts_done - t_txt) if t_tts_done and t_txt else None,
        "llm_speculative": bool(req.meta.get("speculative")) if req else None,
    }
    # sequential decomposition of end_lag_arrival (components sum to it when all present)
    if None not in (t_off, t_eot, t_txt, t_aud, t_last_audio):
        d["stt_finalize_ms"] = _ms(t_eot - t_off)
        d["translate_wait_ms"] = _ms(t_txt - t_eot)
        d["tts_first_byte_ms"] = _ms(t_aud - t_txt)
        d["tts_stream_rest_ms"] = _ms(t_last_audio - t_aud)
    return d


# ---------------------------------------------------------------- EVS
def evs_values(src_word_t_ns: list[int], tgt_word_t_ns: list[int], pairs: list[tuple[int, int]]) -> list[float]:
    """Ear-voice span per aligned (source word, target word) pair, in ms:
    playback time of the target word start minus the time the source word was sent."""
    vals = []
    for i, j in pairs:
        if 0 <= i < len(src_word_t_ns) and 0 <= j < len(tgt_word_t_ns):
            vals.append((tgt_word_t_ns[j] - src_word_t_ns[i]) / NS_PER_MS)
    return vals


# ---------------------------------------------------------------- aggregation
def percentile(values: list[float], q: float) -> float | None:
    v = [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.percentile(v, q)) if v else None


def describe(values: list[float | None]) -> dict[str, Any]:
    v = [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return {"median": None, "p90": None, "min": None, "max": None, "count": 0}
    a = np.asarray(v, dtype=float)
    return {
        "median": round(float(np.median(a)), 2),
        "p90": round(float(np.percentile(a, 90)), 2),
        "min": round(float(a.min()), 2),
        "max": round(float(a.max()), 2),
        "count": int(a.size),
    }
