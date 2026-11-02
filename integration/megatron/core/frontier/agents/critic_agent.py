from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Critique:
    mechanism_confirmed: bool
    explanation: str
    corrected_gain_us: float


class CriticAgent:
    agent_id = "critic"

    def critique(self, prediction, measurement) -> Critique:
        gain = prediction.step_time_us - measurement.step_time_us
        return Critique(gain > 0.0, "measured step time compared with prediction", gain)
