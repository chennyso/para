"""
RegionProfiler: measures forward/backward time and peak memory for one
RegionContract by running it in isolation for warmup + measured steps.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from megatron.core.frontier.profiling.database import ProfileDatabase


@dataclass
class ProfilingResult:
    forward_us_p50: float
    backward_us_p50: float
    peak_memory_bytes: int
    forward_us_p10: float
    forward_us_p90: float


def _time_us() -> float:
    return time.perf_counter() * 1e6


def profile_callable(
    fwd_fn: Callable,
    bwd_fn: Callable | None,
    warmup: int = 10,
    measured: int = 50,
) -> ProfilingResult:
    """
    Profile a forward (and optionally backward) function.
    Uses torch.cuda.synchronize() if torch is available for accurate timing.
    """
    try:
        import torch
        sync = torch.cuda.synchronize
    except ImportError:
        sync = lambda: None  # noqa: E731

    # Warmup
    for _ in range(warmup):
        fwd_fn()
        sync()

    # Measure forward
    fwd_times = []
    for _ in range(measured):
        sync()
        t0 = _time_us()
        fwd_fn()
        sync()
        fwd_times.append(_time_us() - t0)

    fwd_times.sort()

    # Measure backward
    bwd_times = [0.0] * measured
    if bwd_fn is not None:
        for i in range(measured):
            sync()
            t0 = _time_us()
            bwd_fn()
            sync()
            bwd_times[i] = _time_us() - t0
        bwd_times.sort()

    # Peak memory
    peak_mem = 0
    try:
        import torch
        peak_mem = torch.cuda.max_memory_allocated()
    except Exception:
        pass

    n = len(fwd_times)
    return ProfilingResult(
        forward_us_p50=statistics.median(fwd_times),
        backward_us_p50=statistics.median(bwd_times) if bwd_fn else 0.0,
        peak_memory_bytes=peak_mem,
        forward_us_p10=fwd_times[max(0, int(0.1 * n) - 1)],
        forward_us_p90=fwd_times[min(n - 1, int(0.9 * n))],
    )


class RegionProfiler:
    """
    High-level interface: profile a RegionContract and store results in the DB.
    """

    def __init__(self, db: "ProfileDatabase", config_kwargs: dict | None = None):
        self.db = db
        self.config_kwargs = config_kwargs or {}

    def profile_contract(
        self,
        profile_id: str,
        fwd_fn: Callable,
        bwd_fn: Callable | None = None,
        warmup: int = 10,
        measured: int = 50,
        **meta,
    ) -> ProfilingResult:
        result = profile_callable(fwd_fn, bwd_fn, warmup=warmup, measured=measured)
        self.db.insert_region(
            profile_id=profile_id,
            forward_us=result.forward_us_p50,
            backward_us=result.backward_us_p50,
            peak_memory_bytes=result.peak_memory_bytes,
            fwd_p10=result.forward_us_p10,
            fwd_p90=result.forward_us_p90,
            **{k: v for k, v in meta.items() if k in (
                "gpu", "dtype", "seq_len", "micro_batch", "hidden",
                "tp", "cp", "ep", "pp",
            )},
        )
        return result
