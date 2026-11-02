from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path


class AdmissionStatus(Enum):
    PASS = "PASS"
    MEMORY_REJECT_ANALYTIC = "MEMORY_REJECT_ANALYTIC"
    MEMORY_REJECT_INIT = "MEMORY_REJECT_INIT"
    MEMORY_REJECT_RUNTIME = "MEMORY_REJECT_RUNTIME"
    INVALID_CONTRACT = "INVALID_CONTRACT"
    INVALID_GROUP_MAPPING = "INVALID_GROUP_MAPPING"
    COMPILE_FAILURE = "COMPILE_FAILURE"
    RUNTIME_FAILURE = "RUNTIME_FAILURE"
    NUMERICAL_FAILURE = "NUMERICAL_FAILURE"
    KNOWN_MEMORY_FAILURE = "KNOWN_MEMORY_FAILURE"


@dataclass(frozen=True)
class MemoryBreakdown:
    parameter_bytes: int
    gradient_bytes: int
    optimizer_bytes: int
    activation_lower_bound_bytes: int
    communication_bytes: int
    boundary_bytes: int
    runtime_reserve_bytes: int

    @property
    def predicted_peak_bytes(self) -> int:
        return sum(asdict(self).values())


@dataclass(frozen=True)
class MemoryPrediction:
    run_id: str
    gpu_memory_bytes: int
    breakdown: MemoryBreakdown
    stage_peaks: tuple[int, ...]

    @property
    def predicted_peak_bytes(self) -> int:
        return max(self.stage_peaks, default=0)

    @property
    def admitted(self) -> bool:
        return all(peak <= self.gpu_memory_bytes for peak in self.stage_peaks)


@dataclass(frozen=True)
class StageMemoryOwnership:
    """一个 PP stage 在单张 GPU 上拥有的显存组成。"""

    stage: int
    parameter_bytes: int
    gradient_bytes: int
    optimizer_bytes: int
    embedding_head_bytes: int
    moe_expert_bytes: int
    moe_router_bytes: int
    activation_lower_bound_bytes: int
    communication_bytes: int
    boundary_bytes: int

    def __post_init__(self) -> None:
        values = (
            self.stage,
            self.parameter_bytes,
            self.gradient_bytes,
            self.optimizer_bytes,
            self.embedding_head_bytes,
            self.moe_expert_bytes,
            self.moe_router_bytes,
            self.activation_lower_bound_bytes,
            self.communication_bytes,
            self.boundary_bytes,
        )
        if self.stage < 0 or any(value < 0 for value in values[1:]):
            raise ValueError("stage ownership values must be non-negative")

    @property
    def static_parameter_bytes(self) -> int:
        return self.parameter_bytes + self.embedding_head_bytes + self.moe_expert_bytes + self.moe_router_bytes

    def to_breakdown(self, reserve_bytes: int) -> MemoryBreakdown:
        return MemoryBreakdown(
            self.static_parameter_bytes,
            self.gradient_bytes,
            self.optimizer_bytes,
            self.activation_lower_bound_bytes,
            self.communication_bytes,
            self.boundary_bytes,
            reserve_bytes,
        )


@dataclass(frozen=True)
class StageMemoryPrediction:
    run_id: str
    gpu_memory_bytes: int
    reserve_bytes: int
    stages: tuple[StageMemoryOwnership, ...]

    @property
    def stage_peaks(self) -> tuple[int, ...]:
        return tuple(stage.to_breakdown(self.reserve_bytes).predicted_peak_bytes for stage in self.stages)

    @property
    def predicted_peak_bytes(self) -> int:
        return max(self.stage_peaks, default=0)

    @property
    def admitted(self) -> bool:
        return all(peak <= self.gpu_memory_bytes for peak in self.stage_peaks)

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "gpu_memory_bytes": self.gpu_memory_bytes,
            "reserve_bytes": self.reserve_bytes,
            "predicted_peak_bytes": self.predicted_peak_bytes,
            "admitted": self.admitted,
            "stages": [
                {
                    "stage": stage.stage,
                    "parameter_bytes": stage.parameter_bytes,
                    "gradient_bytes": stage.gradient_bytes,
                    "optimizer_bytes": stage.optimizer_bytes,
                    "embedding_head_bytes": stage.embedding_head_bytes,
                    "moe_expert_bytes": stage.moe_expert_bytes,
                    "moe_router_bytes": stage.moe_router_bytes,
                    "activation_lower_bound_bytes": stage.activation_lower_bound_bytes,
                    "communication_bytes": stage.communication_bytes,
                    "boundary_bytes": stage.boundary_bytes,
                    "predicted_total_bytes": stage.to_breakdown(self.reserve_bytes).predicted_peak_bytes,
                }
                for stage in self.stages
            ],
        }


@dataclass(frozen=True)
class AdmissionRecord:
    run_id: str
    status: AdmissionStatus
    predicted_peak_bytes: int
    actual_peak_bytes: int | None = None
    reason: str = ""
    metadata: dict | None = None


class MemoryAdmission:
    def __init__(self, gpu_memory_bytes: int = 32 * 1024**3, reserve_bytes: int | None = None) -> None:
        if gpu_memory_bytes <= 0:
            raise ValueError("GPU memory must be positive")
        minimum_reserve = 4 * 1024**3
        percentage_reserve = (gpu_memory_bytes * 125 + 999) // 1000
        self.gpu_memory_bytes = gpu_memory_bytes
        self.reserve_bytes = max(minimum_reserve, percentage_reserve) if reserve_bytes is None else reserve_bytes
        if self.reserve_bytes <= 0 or self.reserve_bytes >= gpu_memory_bytes:
            raise ValueError("memory reserve must be positive and below GPU memory")

    def predict(self, run_id: str, parameter_bytes: int, gradient_bytes: int, optimizer_bytes: int, activation_lower_bound_bytes: int, communication_bytes: int, boundary_bytes: int, stage_count: int) -> MemoryPrediction:
        values = (parameter_bytes, gradient_bytes, optimizer_bytes, activation_lower_bound_bytes, communication_bytes, boundary_bytes)
        if stage_count < 1 or any(value < 0 for value in values):
            raise ValueError("memory terms must be non-negative and stage count must be positive")
        breakdown = MemoryBreakdown(*values, self.reserve_bytes)
        stage_peak = breakdown.predicted_peak_bytes
        return MemoryPrediction(run_id, self.gpu_memory_bytes, breakdown, tuple(stage_peak for _ in range(stage_count)))

    def analytic(self, prediction: MemoryPrediction) -> AdmissionRecord:
        if prediction.admitted:
            return AdmissionRecord(prediction.run_id, AdmissionStatus.PASS, max(prediction.stage_peaks))
        return AdmissionRecord(prediction.run_id, AdmissionStatus.MEMORY_REJECT_ANALYTIC, max(prediction.stage_peaks), reason="predicted peak exceeds GPU memory")

    def predict_stage_ownership(self, run_id: str, stages: tuple[StageMemoryOwnership, ...]) -> StageMemoryPrediction:
        if not stages:
            raise ValueError("at least one PP stage is required")
        expected = tuple(range(len(stages)))
        actual = tuple(stage.stage for stage in stages)
        if actual != expected:
            raise ValueError("stage ownership must be ordered and contiguous")
        return StageMemoryPrediction(run_id, self.gpu_memory_bytes, self.reserve_bytes, stages)

    def analytic_stage_ownership(self, prediction: StageMemoryPrediction) -> AdmissionRecord:
        if prediction.admitted:
            return AdmissionRecord(prediction.run_id, AdmissionStatus.PASS, prediction.predicted_peak_bytes)
        return AdmissionRecord(
            prediction.run_id,
            AdmissionStatus.MEMORY_REJECT_ANALYTIC,
            prediction.predicted_peak_bytes,
            reason="predicted PP stage peak exceeds GPU memory",
            metadata={"stage_peaks": list(prediction.stage_peaks)},
        )

    def record_probe(self, prediction: MemoryPrediction | StageMemoryPrediction, stage: str, peak_bytes: int, completed_steps: int) -> AdmissionRecord:
        if peak_bytes < 0 or completed_steps < 0:
            raise ValueError("probe measurements must be non-negative")
        if stage not in {"initialization", "runtime"}:
            raise ValueError("probe stage must be initialization or runtime")
        if stage == "runtime" and completed_steps < 2:
            raise ValueError("runtime probe requires two completed steps")
        if peak_bytes >= self.gpu_memory_bytes:
            status = AdmissionStatus.MEMORY_REJECT_INIT if stage == "initialization" else AdmissionStatus.MEMORY_REJECT_RUNTIME
            return AdmissionRecord(prediction.run_id, status, max(prediction.stage_peaks), peak_bytes, "measured peak reached GPU memory")
        return AdmissionRecord(prediction.run_id, AdmissionStatus.PASS, max(prediction.stage_peaks), peak_bytes, metadata={"probe_stage": stage, "completed_steps": completed_steps})


class AdmissionLedger:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, list):
            raise ValueError("admission ledger must contain a list")
        return value

    def record(self, record: AdmissionRecord) -> None:
        current = self._read()
        previous = next((item for item in current if item.get("run_id") == record.run_id), None)
        records = [item for item in current if item.get("run_id") != record.run_id]
        value = asdict(record)
        value["status"] = record.status.value
        if previous is not None:
            if value["predicted_peak_bytes"] == 0:
                value["predicted_peak_bytes"] = previous.get("predicted_peak_bytes", 0)
            if previous.get("metadata") and value.get("metadata"):
                value["metadata"] = {**previous["metadata"], **value["metadata"]}
            elif previous.get("metadata") and value.get("metadata") is None:
                value["metadata"] = previous["metadata"]
        records.append(value)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2, sort_keys=True)

    def status(self, run_id: str) -> AdmissionStatus | None:
        for item in self._read():
            if item.get("run_id") == run_id:
                return AdmissionStatus(item["status"])
        return None

    def reject_duplicate(self, run_id: str) -> AdmissionRecord | None:
        status = self.status(run_id)
        if status is None:
            return None
        if status is AdmissionStatus.PASS:
            raise ValueError(f"run already admitted: {run_id}")
        return AdmissionRecord(run_id, status, 0, reason="run already has a recorded rejection")

    def record_known_failure(self, run_id: str, reason: str, metadata: dict | None = None) -> AdmissionRecord:
        record = AdmissionRecord(
            run_id,
            AdmissionStatus.KNOWN_MEMORY_FAILURE,
            0,
            reason=reason,
            metadata=metadata,
        )
        self.record(record)
        return record
