"""One clock for every timestamp: time.perf_counter_ns().

Also provides an accurate async `sleep_until_ns` and a paced sender driven by an
absolute monotonic schedule (chunk i is due at t_start + i * chunk_ns), so per-chunk
scheduling error never accumulates into drift.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

now_ns = time.perf_counter_ns

_SPIN_NS = 300_000  # finish the last 0.3 ms of every wait by yielding in a tight loop
_hires_enabled = False
# Coarse waits run time.sleep() in these threads: on Windows (CPython >= 3.11) that is a
# high-resolution waitable timer (~0.5 ms), unlike asyncio's loop timers, which follow the
# 15.6 ms system tick whenever Windows declines a timeBeginPeriod request (it may for
# background processes). Measured on the dev laptop: loop timers p99 15 ms / max 43 ms
# late, thread timers p99 1.5 ms / max 3.6 ms.
_TIMER_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=16, thread_name_prefix="bench-timer")


def enable_high_res_timer() -> bool:
    """On Windows, request 1 ms system timer resolution for this process.

    Without it, asyncio sleeps on Windows overshoot by ~10-15 ms (15.6 ms timer tick).
    Since Windows 10 2004 the request is scoped to this process and released on exit.
    Returns True if a change was made.
    """
    global _hires_enabled
    if _hires_enabled or sys.platform != "win32":
        return False
    import ctypes

    winmm = ctypes.WinDLL("winmm")
    if winmm.timeBeginPeriod(1) == 0:
        _hires_enabled = True
        atexit.register(winmm.timeEndPeriod, 1)
        return True
    return False


def _precise_loop_factory() -> asyncio.AbstractEventLoop:
    """Event loop whose scheduler clock is perf_counter, not time.monotonic.

    On Windows/CPython 3.11, time.monotonic() is GetTickCount64 (15.6 ms resolution) and
    asyncio's _clock_resolution is 15.6 ms, so timer deadlines are quantized and sleeps
    run late by up to a tick whenever several timers interleave. perf_counter is
    monotonic too (QueryPerformanceCounter), so every library timeout keeps working.
    """
    base = asyncio.ProactorEventLoop if sys.platform == "win32" else asyncio.SelectorEventLoop

    class PreciseLoop(base):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__()
            self._clock_resolution = time.get_clock_info("perf_counter").resolution

        def time(self) -> float:
            return time.perf_counter()

    return PreciseLoop()


def raise_process_priority() -> None:
    """Ask the OS to schedule this process ahead of normal background work (this process
    only; reverts when it exits). Reduces sporadic preemption stalls in pacing."""
    try:
        if sys.platform == "win32":
            import ctypes

            k32 = ctypes.WinDLL("kernel32")
            k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00008000)  # ABOVE_NORMAL_PRIORITY_CLASS
        else:
            import os

            os.nice(-5)
    except Exception:
        pass  # unprivileged on Unix: fine, keep default priority


def keep_system_awake() -> None:
    """Ask Windows not to idle-sleep or enter standby while this process runs
    (SetThreadExecutionState; a per-process request released on exit, not a settings change).
    Round 2 lost four runs when the laptop entered Modern Standby mid-batch on battery."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_AWAYMODE_REQUIRED = 0x80000000, 0x00000001, 0x00000040
        ctypes.WinDLL("kernel32").SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED)
    except Exception:
        pass


def run(coro):  # noqa: ANN001, ANN201
    """asyncio.run() on the precise loop with a 1 ms OS timer. Use for every measurement."""
    enable_high_res_timer()
    raise_process_priority()
    keep_system_awake()
    # Hand the GIL over faster (default 5 ms): a timer thread whose sleep has finished must
    # take the GIL to wake the loop; with the default interval it can wait behind a
    # spinning loop for several ms. Measured: max send lateness 73 ms -> ~20 ms.
    sys.setswitchinterval(0.0005)
    with asyncio.Runner(loop_factory=_precise_loop_factory) as runner:
        return runner.run(coro)


async def sleep_until_ns(target_ns: int) -> None:
    """Sleep until perf_counter_ns() >= target_ns.

    Coarse wait on a high-resolution timer thread until ~0.3 ms before the target, then
    yield-spin with asyncio.sleep(0) so other tasks (receivers stamping arrivals) keep
    running while we close the last millisecond.
    """
    remaining = target_ns - now_ns()
    if remaining > _SPIN_NS:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(_TIMER_POOL, time.sleep, (remaining - _SPIN_NS) / 1e9)
    while now_ns() < target_ns:
        await asyncio.sleep(0)


@dataclass
class SendRecord:
    planned_ns: list[int] = field(default_factory=list)
    sent_ns: list[int] = field(default_factory=list)

    @property
    def max_lateness_ms(self) -> float:
        if not self.sent_ns:
            return 0.0
        return max(s - p for s, p in zip(self.sent_ns, self.planned_ns)) / 1e6

    @property
    def p99_lateness_ms(self) -> float:
        if not self.sent_ns:
            return 0.0
        late = sorted(s - p for s, p in zip(self.sent_ns, self.planned_ns))
        return late[min(len(late) - 1, int(0.99 * len(late)))] / 1e6

    @property
    def drift_ms(self) -> float:
        """Systematic drift: median send error over the last 50 chunks minus over the
        first 50 (robust to a single preempted chunk at either end)."""
        n = len(self.sent_ns)
        if n < 2:
            return 0.0
        k = max(1, min(50, n // 4))
        err = [s - p for s, p in zip(self.sent_ns, self.planned_ns)]
        med = lambda v: sorted(v)[len(v) // 2]  # noqa: E731
        return (med(err[-k:]) - med(err[:k])) / 1e6


async def paced_send(
    chunks: Sequence[bytes],
    chunk_ns: int,
    send: Callable[[int, bytes, int], Awaitable[None]],
    start_delay_ns: int = 5_000_000,
    record: SendRecord | None = None,
) -> SendRecord:
    """Send chunks at exactly 1x real time.

    `send(index, chunk, t_sent_ns)` is awaited for each chunk; t_sent_ns is stamped
    immediately before the call. If a send runs late (backpressure), later chunks are
    still scheduled against the original absolute timeline, so the stream catches up
    instead of drifting.
    """
    rec = record if record is not None else SendRecord()
    t_start = now_ns() + start_delay_ns
    for i, chunk in enumerate(chunks):
        due = t_start + i * chunk_ns
        await sleep_until_ns(due)
        t = now_ns()
        rec.planned_ns.append(due)
        rec.sent_ns.append(t)
        await send(i, chunk, t)
    return rec
