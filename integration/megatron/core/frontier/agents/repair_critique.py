from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ConflictRepairAdvice(BaseModel):
    model_config = ConfigDict(extra='forbid')
    conflict_id: str = Field(min_length=1)
    explanation: str = Field(min_length=1)
    responsible_agent: Literal['Architect', 'Program']
    requested_changes: list[str] = Field(min_length=1)


class RepairCritique(BaseModel):
    model_config = ConfigDict(extra='forbid')
    program_id: str = Field(min_length=1)
    advice: list[ConflictRepairAdvice] = Field(min_length=1)
