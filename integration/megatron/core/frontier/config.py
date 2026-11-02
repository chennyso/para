from __future__ import annotations
"""Global configuration for the Frontier planning subsystem."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FrontierConfig:
    """Top-level knobs for Frontier planning and search."""

    # Planner beam width — number of partial plans kept at each region step.
    beam_width: int = 32

    # How often (in regions) to invoke the full simulator during beam search.
    # Between these checkpoints a cheaper profile-based lower bound is used.
    simulator_interval: int = 3

    # Maximum number of real training executions the agent may trigger.
    agent_execution_budget: int = 50

    # Whether to emit NVTX ranges in the training hot path for Nsight profiling.
    nvtx_enabled: bool = True

    # SQLite path for the profile database.
    profile_db: str = "profiles/frontier.db"

    # Minimum warmup steps used when measuring a configuration.
    profile_warmup: int = 10

    # Number of measured steps per profiling run.
    profile_measured: int = 50

    # Number of independent profiling jobs per configuration.
    profile_repeats: int = 3

    # Absolute memory budget per GPU in bytes.  None means "use device maximum".
    memory_budget_bytes: int | None = None

    # Absolute time budget for the planner (seconds).  None means unlimited.
    planner_time_budget_seconds: float | None = 120.0

    # Dominance comparison dimensions used in the beam.
    dominance_dims: list[str] = field(default_factory=lambda: [
        "max_stage_load_us",
        "peak_memory_bytes",
        "max_ready_time_us",
        "nic_pressure",
        "nvlink_pressure",
    ])

    enabled: bool = False
    plan_path: Optional[str] = None
    trace_enabled: bool = False
    trace_output_dir: str = "frontier_traces"
    segment_width: int = 4
    max_boundary_candidates: int = 5
    memory_reserve_bytes: int = 2 * 1024**3
    agent_enabled: bool = False
    max_real_trials: int = 20
