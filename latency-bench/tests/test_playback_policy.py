"""Catch-up playback policy math on synthetic arrival schedules worked out by hand.

Blocks are 20 ms of content. "Burst" = 10 chunks of 100 ms (50 blocks, 1,000 ms of
content) that all arrive at t = 0.
"""

import numpy as np
import pytest

from bench.analysis.metrics import (
    OutChunk, PlaybackPolicy, block_time_of_sample, policy_end_lag_ms, simulate_playback,
    simulate_playback_policy, utterance_metrics,
)
from bench.events import ConnectRecord, RunData, RunFooter, RunHeader, SendRecordModel, SourceTiming

MS = 1_000_000
SR = 24000


def burst(n=10, dur_ms=100, at_ms=0):
    step = SR * dur_ms // 1000
    return [OutChunk(at_ms * MS, step, SR, i * step, 0, step - 1) for i in range(n)]


def end_ms(blocks):
    return blocks[-1].end_ns / MS


def test_none_policy_matches_round1_simulator():
    rng = np.random.default_rng(1)
    t, chunks, off = 0, [], 0
    for _ in range(40):
        t += int(rng.integers(0, 60)) * MS
        n = int(rng.integers(1, 12)) * 240
        chunks.append(OutChunk(t, n, SR, off, 0, n - 1))
        off += n
    old = simulate_playback(chunks)
    blocks = simulate_playback_policy(chunks, PlaybackPolicy("none"))
    for ci, (s, e) in enumerate(old):
        bs = [b for b in blocks if b.chunk == ci]
        assert bs[0].start_ns == s
        assert abs(bs[-1].end_ns - e) <= len(bs)  # rounding of per-block durations (ns)


def test_burst_catchup_1_25x_threshold_500():
    # backlog at block i = 1000 - 20i ms. Fast from block 0 (1000 > 500) until backlog < 250
    # (block 38: 240 ms). 38 blocks x 16 ms + 12 blocks x 20 ms = 608 + 240 = 848 ms.
    b = simulate_playback_policy(burst(), PlaybackPolicy("catchup", 500, 1.25))
    assert end_ms(b) == pytest.approx(848.0, abs=0.01)
    assert [x.speed for x in b[:38]] == [1.25] * 38 and [x.speed for x in b[38:]] == [1.0] * 12


def test_burst_no_policy_is_1000ms():
    assert end_ms(simulate_playback_policy(burst(), PlaybackPolicy("none"))) == pytest.approx(1000.0, abs=0.01)


def test_speed_is_capped_at_max_speed():
    # requested 2.0x, capped at 1.5x: 38 x 13.333 + 240 = 746.667 ms
    b = simulate_playback_policy(burst(), PlaybackPolicy("catchup", 500, 2.0, max_speed=1.5))
    assert end_ms(b) == pytest.approx(38 * 20 / 1.5 + 240, abs=0.01)


def test_threshold_300_engages_and_releases_differently():
    # fast until backlog < 150 ms: 1000 - 20i < 150 -> i = 43. 43 x 16 + 7 x 20 = 688 + 140 = 828 ms
    b = simulate_playback_policy(burst(), PlaybackPolicy("catchup", 300, 1.25))
    assert end_ms(b) == pytest.approx(828.0, abs=0.01)


def test_steady_stream_below_threshold_never_speeds_up():
    chunks = [OutChunk(i * 20 * MS, 480, SR, i * 480, 0, 479) for i in range(100)]
    b = simulate_playback_policy(chunks, PlaybackPolicy("catchup", 500, 1.4))
    assert all(x.speed == 1.0 for x in b)
    assert end_ms(b) == pytest.approx(2000.0, abs=0.01)


def test_never_starts_before_arrival():
    # 3 x 100 ms at t=0, then 1 x 100 ms arriving at 2,000 ms (long after the queue drained)
    chunks = burst(3) + [OutChunk(2000 * MS, 2400, SR, 3 * 2400, 0, 2399)]
    b = simulate_playback_policy(chunks, PlaybackPolicy("catchup", 100, 1.5))
    assert all(x.start_ns >= x.arrival_ns for x in b)
    assert [x for x in b if x.chunk == 3][0].start_ns == 2000 * MS


def test_catchup_end_only_speeds_inside_window():
    # window opens at 500 ms. Blocks 0-24 play at 1x (0-500 ms). Hysteresis is already
    # engaged, so blocks 25-37 play at 1.25x (13 x 16 = 208), then 12 x 20 = 240 at 1x.
    b = simulate_playback_policy(burst(), PlaybackPolicy("catchup_end", 500, 1.25),
                                 speed_windows=[(500 * MS, 10_000 * MS)])
    assert end_ms(b) == pytest.approx(500 + 208 + 240, abs=0.01)
    assert all(x.speed == 1.0 for x in b[:25])


def test_policy_end_lag_and_sample_time_with_speed():
    chunks = burst()
    b = simulate_playback_policy(chunks, PlaybackPolicy("catchup", 500, 1.25))
    # last audible sample is the last sample: ends at 848 ms; offset at t=100 ms -> 748 ms
    assert policy_end_lag_ms(chunks, b, 100 * MS) == pytest.approx(748.0, abs=0.01)
    # sample 480 (start of block 1, fast) starts at 16 ms
    assert block_time_of_sample(b, 480) == 16 * MS


def _run_with_segments(segments_s, sent_every_ms=20, n_chunks=500):
    sent = [i * sent_every_ms * MS for i in range(n_chunks)]
    h = RunHeader(run_id="r", batch_id="b", provider="p", clip_id="c", pair="es-en", category="two_utterances",
                  synthetic=True, repetition=0, order_index=0, source_lang="es", target_lang="en", chunk_ms=20,
                  input_sample_rate=16000, chunk_samples=320, n_chunks=n_chunks,
                  source=SourceTiming(sample_rate=24000, n_samples=n_chunks * 480, onset_sample=None,
                                      offset_sample=None,
                                      segments=[(int(a * 24000), int(b * 24000)) for a, b in segments_s]))
    return RunData(h, ConnectRecord(t_connect_start_ns=0, t_session_ready_ns=0),
                   SendRecordModel(planned_ns=sent, sent_ns=sent, max_lateness_ms=0, drift_ms=0), [], RunFooter(status="ok"))


def test_multi_utterance_assignment_and_lags():
    # utterance 0: 1.0-3.0 s, utterance 1: 4.5-6.0 s
    run = _run_with_segments([(1.0, 3.0), (4.5, 6.0)])
    chunks = [
        OutChunk(2500 * MS, 2400, SR, 0, 0, 2399),        # before first offset -> utt 0
        OutChunk(3400 * MS, 2400, SR, 2400, 0, 2399),     # after offset 0 -> utt 0
        OutChunk(5000 * MS, 2400, SR, 4800, 0, 2399),     # after offset 0, before offset 1 -> utt 0
        OutChunk(6900 * MS, 4800, SR, 7200, 0, 4799),     # after offset 1 -> utt 1
    ]
    blocks = simulate_playback_policy(chunks, PlaybackPolicy("none"))
    u = utterance_metrics(run, chunks, blocks)
    assert [x["n_chunks"] for x in u] == [3, 1]
    # utt 0 last chunk plays 5000-5100 ms -> end lag 5100 - 3000 = 2100
    assert u[0]["end_lag_playback_ms"] == pytest.approx(2100.0, abs=0.01)
    # utt 1 chunk plays 6900-7100 -> end lag 7100 - 6000 = 1100
    assert u[1]["end_lag_playback_ms"] == pytest.approx(1100.0, abs=0.01)
    # start lag: first audible arrival after onset: utt0 2500-1000; utt1 5000-4500
    assert u[0]["start_lag_ms"] == 1500.0 and u[1]["start_lag_ms"] == 500.0


def test_parallel_tracks_are_mixed_not_concatenated():
    # two sessions each stream 20 ms chunks every 20 ms for 1 s (continuous real-time filler + speech)
    from bench.analysis.metrics import simulate_playback_tracks

    chunks, tracks, off = [], [], 0
    for i in range(50):
        for s in (0, 1):
            chunks.append(OutChunk(i * 20 * MS, 480, SR, off, 0, 479))
            tracks.append(s)
            off += 480
    concat = simulate_playback_policy(chunks, PlaybackPolicy("none"))
    assert end_ms(concat) == pytest.approx(2000.0, abs=0.01)       # concatenation doubles: wrong
    mixed, overlap = simulate_playback_tracks(chunks, tracks)
    assert max(b.end_ns for b in mixed) / MS == pytest.approx(1000.0, abs=0.01)  # each track real time
    assert overlap == pytest.approx(1.0, abs=0.001)                  # both audible the whole second
    assert sorted(b.chunk for b in mixed) == list(range(len(chunks)))
