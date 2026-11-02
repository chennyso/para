"""ContentionProfiler: measures I_ij = T(c_i || c_j) / T0(c_i)."""
from __future__ import annotations

import statistics
import time
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from megatron.core.frontier.profiling.database import ProfileDatabase


def measure_contention(
    fn_a: Callable,
    fn_b: Callable,
    measured: int = 30,
) -> float:
    """
    Run fn_a alone and fn_a+fn_b concurrently (in threads).
    Return I = T(a||b) / T0(a).
    """
    import threading

    def _time_us():
        return time.perf_counter() * 1e6

    try:
        import torch
        sync = torch.cuda.synchronize
    except ImportError:
        sync = lambda: None  # noqa: E731

    # Baseline: fn_a alone
    alone_times = []
    for _ in range(measured):
        sync()
        t0 = _time_us()
        fn_a()
        sync()
        alone_times.append(_time_us() - t0)

    # Concurrent
    concurrent_times = []
    for _ in range(measured):
        sync()
        t0 = _time_us()
        t = threading.Thread(target=fn_b)
        t.start()
        fn_a()
        t.join()
        sync()
        concurrent_times.append(_time_us() - t0)

    t0_median = statistics.median(alone_times)
    tc_median = statistics.median(concurrent_times)
    return tc_median / max(t0_median, 1.0)


class ContentionProfiler:
    def __init__(self, db: "ProfileDatabase"):
        self.db = db

    def profile_pair(
        self,
        resource_a: str,
        resource_b: str,
        fn_a: Callable,
        fn_b: Callable,
        measured: int = 30,
    ) -> float:
        factor = measure_contention(fn_a, fn_b, measured=measured)
        self.db.insert_contention(resource_a, resource_b, factor)
        return factor
