import asyncio

from bench import clock
from bench.clock import SendRecord, now_ns, paced_send, sleep_until_ns


def test_paced_sender_drift_under_2ms_over_30s():
    """30 s of 20 ms chunks: the stream must not drift from the absolute schedule."""
    n = 30 * 50
    t_first_last: list[int] = []

    async def send(i: int, chunk: bytes, t: int) -> None:
        if i in (0, n - 1):
            t_first_last.append(t)

    rec: SendRecord = clock.run(paced_send([b""] * n, 20_000_000, send))
    assert len(rec.sent_ns) == n
    # systematic drift (robust): median error at the end vs at the start
    assert abs(rec.drift_ms) < 2.0, rec.drift_ms
    # elapsed time first->last send vs ideal 29.98 s (single-sample, so allow one preemption)
    elapsed_err_ms = (t_first_last[1] - t_first_last[0]) / 1e6 - (n - 1) * 20
    assert abs(elapsed_err_ms) < 20.0, elapsed_err_ms
    # sends stay on schedule; at most a tiny fraction delayed by OS preemption
    assert rec.p99_lateness_ms < 5.0, rec.p99_lateness_ms
    assert min(s - p for s, p in zip(rec.sent_ns, rec.planned_ns)) >= 0  # never early


def test_sleep_until_is_accurate_and_never_early():
    async def go() -> list[float]:
        errs = []
        for _ in range(100):
            target = now_ns() + 7_000_000
            await sleep_until_ns(target)
            errs.append((now_ns() - target) / 1e6)
        return errs

    errs = sorted(clock.run(go()))
    assert errs[0] >= 0
    assert errs[len(errs) // 2] < 1.0
    assert errs[-2] < 10.0


def test_late_send_catches_up_instead_of_drifting():
    """A slow send must not shift the rest of the schedule."""
    async def send(i: int, chunk: bytes, t: int) -> None:
        if i == 10:
            await asyncio.sleep(0.1)  # 100 ms stall

    rec = clock.run(paced_send([b""] * 50, 20_000_000, send))
    late = [(s - p) / 1e6 for s, p in zip(rec.sent_ns, rec.planned_ns)]
    assert late[11] > 50          # immediately after the stall we are behind...
    assert late[-1] < 5           # ...and back on the original timeline by the end
