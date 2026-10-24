"""Metric functions on synthetic, hand-computable event streams."""

import numpy as np
import pytest

from bench.analysis.metrics import (
    OutChunk, audible_bounds, cascade_stages, chunk_of_sample, describe, evs_values, latency_metrics,
    output_chunks, playback_time_of_output_sample, precise_time_of_sample, send_time_of_sample,
    simulate_playback,
)
from bench.events import ConnectRecord, EventRecord, RunData, RunFooter, RunHeader, SendRecordModel, SourceTiming

MS = 1_000_000
SR_IN = 16000
SR_OUT = 24000
CHUNK_MS = 20


def make_run(onset_s: float, offset_s: float, audio: list[tuple[float, int, bool]], n_chunks: int = 300,
             extra_events: list[EventRecord] | None = None, t0: int = 10_000 * MS) -> tuple[RunData, np.ndarray]:
    """audio: (arrival_ms_after_t0, n_samples_at_24k, audible). Sends are exactly every 20 ms from t0."""
    sent = [t0 + i * 20 * MS for i in range(n_chunks)]
    header = RunHeader(
        run_id="r", batch_id="b", provider="p", clip_id="c", pair="es-en", category="short", synthetic=True,
        repetition=0, order_index=0, source_lang="es", target_lang="en", chunk_ms=CHUNK_MS,
        input_sample_rate=SR_IN, chunk_samples=320, n_chunks=n_chunks,
        source=SourceTiming(sample_rate=SR_IN, n_samples=n_chunks * 320,
                            onset_sample=int(onset_s * SR_IN), offset_sample=int(offset_s * SR_IN)),
    )
    events, pcm_parts, off = [], [], 0
    for arr_ms, n, loud in audio:
        x = np.full(n, 8000 if loud else 3, dtype=np.int16)
        events.append(EventRecord(kind="audio_out", t_recv_ns=t0 + int(arr_ms * MS), sample_rate=SR_OUT,
                                  audio_offset=off, n_samples=n))
        pcm_parts.append(x)
        off += n
    events += extra_events or []
    events.sort(key=lambda e: e.t_recv_ns)
    run = RunData(header, ConnectRecord(t_connect_start_ns=t0 - 500 * MS, t_session_ready_ns=t0 - 200 * MS),
                  SendRecordModel(planned_ns=sent, sent_ns=sent, max_lateness_ms=0, drift_ms=0),
                  events, RunFooter(status="ok"))
    return run, (np.concatenate(pcm_parts) if pcm_parts else np.zeros(0, np.int16))


def test_sample_to_send_time_mapping():
    sent = [1000 + i * 20 * MS for i in range(10)]
    assert chunk_of_sample(0, SR_IN, 20) == 0
    assert chunk_of_sample(319, SR_IN, 20) == 0
    assert chunk_of_sample(320, SR_IN, 20) == 1
    assert send_time_of_sample(sent, 330, SR_IN, 20) == sent[1]
    # precise: 10 samples into chunk 1 = 0.625 ms after its send
    assert precise_time_of_sample(sent, 330, SR_IN, 20) == sent[1] + 625_000


def test_audible_bounds_gate():
    x = np.array([0, 5, 10, 3000, 20, 4000, 2, 0], dtype=np.int16)
    assert audible_bounds(x, -45.0) == (3, 5)        # -45 dBFS ~ 184 counts
    assert audible_bounds(np.zeros(10, np.int16), -45.0) == (None, None)


def test_start_and_end_lag_and_setup():
    # onset at 1.000 s -> chunk 50 sent at +1000 ms ; offset 2.010 s -> chunk 100 sent at +2000 ms
    audio = [(1250, 480, False), (1300, 480, True), (1320, 480, True), (2350, 480, True), (2370, 480, False)]
    run, pcm = make_run(1.0, 2.010, audio)
    m = latency_metrics(run, output_chunks(run, pcm, -45.0))
    assert m["setup_ms"] == 300.0
    assert m["first_byte_lag_ms"] == 250.0       # silent leading frame counts for first byte...
    assert m["start_lag_ms"] == 300.0            # ...but not for start lag
    assert m["end_lag_arrival_ms"] == 350.0      # last AUDIBLE chunk at 2350 - 2000
    assert m["produced_output"] is True


def test_playback_simulation_queues_back_to_back():
    # three 20 ms chunks arrive in a burst at t=0; a 4th arrives late at t=100
    chunks = [OutChunk(0, 480, SR_OUT, 0, 0, 479), OutChunk(0, 480, SR_OUT, 480, 0, 479),
              OutChunk(5 * MS, 480, SR_OUT, 960, 0, 479), OutChunk(100 * MS, 480, SR_OUT, 1440, 0, 479)]
    spans = simulate_playback(chunks)
    assert spans == [(0, 20 * MS), (20 * MS, 40 * MS), (40 * MS, 60 * MS), (100 * MS, 120 * MS)]
    assert playback_time_of_output_sample(chunks, spans, 480 + 240) == 30 * MS


def test_end_lag_playback_includes_queueing_and_last_audible_sample():
    # speech offset at 2.0 s (chunk 100 sent at +2000). 10 chunks of 100 ms arrive in a burst at +2300:
    # playback ends at 2300 + 10*100 = 3300 -> end lag playback = 1300 ms; arrival = 300 ms.
    audio = [(2300, 2400, True)] * 10
    run, pcm = make_run(1.0, 2.0, audio)
    m = latency_metrics(run, output_chunks(run, pcm, -45.0))
    assert m["end_lag_arrival_ms"] == 300.0
    assert m["end_lag_playback_ms"] == pytest.approx(1300.0, abs=0.01)


def test_end_lag_playback_stops_at_last_audible_sample():
    # final chunk: first half loud, second half silent -> playback end counts only the loud half
    run, _ = make_run(1.0, 2.0, [(2300, 480, True)])
    pcm = np.concatenate([np.full(240, 8000, np.int16), np.zeros(240, np.int16)])
    m = latency_metrics(run, output_chunks(run, pcm, -45.0))
    assert m["end_lag_playback_ms"] == pytest.approx(310.0, abs=0.01)


def test_no_output_for_silence_clip():
    run, pcm = make_run(1.0, 2.0, [(500, 480, False)])
    m = latency_metrics(run, output_chunks(run, pcm, -45.0))
    assert m["produced_output"] is False
    assert m["start_lag_ms"] is None and m["end_lag_playback_ms"] is None


def test_cascade_stage_decomposition_sums_to_end_lag():
    def mark(ms, stage, **meta):
        return EventRecord(kind="stage_mark", t_recv_ns=10_000 * MS + ms * MS, stage=stage, meta=meta)

    marks = [
        mark(2400, "stt.end_of_turn", turn=0),
        mark(2410, "llm.request", seg=0, turn=0),
        mark(2600, "llm.first_token", seg=0),
        mark(2605, "tts.first_text", seg=0),
        mark(2900, "tts.first_audio", seg=0),
        mark(3000, "llm.done", seg=0),
        mark(3400, "tts.done", seg=0),
    ]
    run, pcm = make_run(1.0, 2.0, [(2900, 480, True), (3300, 480, True)], extra_events=marks)
    chunks = output_chunks(run, pcm, -45.0)
    for c in chunks:
        c.seg = 0
    d = cascade_stages(run, chunks)
    assert d["stt_finalize_ms"] == 400.0
    assert d["translate_wait_ms"] == 205.0
    assert d["tts_first_byte_ms"] == 295.0
    assert d["tts_stream_rest_ms"] == 400.0
    total = sum(d[k] for k in ("stt_finalize_ms", "translate_wait_ms", "tts_first_byte_ms", "tts_stream_rest_ms"))
    assert total == latency_metrics(run, chunks)["end_lag_arrival_ms"]
    assert d["llm_first_token_ms"] == 190.0 and d["llm_complete_ms"] == 590.0


def test_evs_values():
    src = [0, 100 * MS, 400 * MS]
    tgt = [900 * MS, 1200 * MS]
    assert evs_values(src, tgt, [(0, 0), (2, 1), (5, 0)]) == [900.0, 800.0]


def test_describe():
    d = describe([1, 2, 3, 4, 100, None])
    assert d["median"] == 3 and d["min"] == 1 and d["max"] == 100 and d["count"] == 5
    assert d["p90"] == pytest.approx(61.6)
    assert describe([None])["count"] == 0
