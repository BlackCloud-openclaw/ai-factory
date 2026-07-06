# experiments/phase1/planning_contract_local.py
"""
Planning Contract v1.0 - 本地副本（独立于 src/writing）
仅供实验使用，避免循环导入
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class ExecutionUnit(BaseModel):
    id: str
    label: Literal["action", "beat", "intent", "constraint"]
    description: str
    attributes: Dict[str, Any] = Field(default_factory=dict)


class Execution(BaseModel):
    units: List[ExecutionUnit] = Field(default_factory=list)


class Constraint(BaseModel):
    type: Literal["required", "forbidden", "before", "after", "exclusive", "at_least_once"]
    target: str
    condition: Optional[str] = None
    refs: Optional[List[str]] = None

    @field_validator("target")
    @classmethod
    def validate_target(cls, v: str) -> str:
        if not v or len(v.strip()) < 2:
            raise ValueError("约束目标至少需要2个字符")
        return v.strip()


class StateChange(BaseModel):
    type: Literal["plot_flag", "relationship", "inventory", "realm", "location", "hp"]
    name: Optional[str] = None
    value: Optional[Any] = None
    from_char: Optional[str] = None
    to_char: Optional[str] = None
    delta: Optional[int] = None
    actor: Optional[str] = None
    item: Optional[str] = None
    operation: Optional[Literal["acquire", "lose"]] = None
    quantity: Optional[int] = 1
    to_major_realm: Optional[str] = None
    to_minor_stage: Optional[int] = None
    location: Optional[str] = None
    new_hp: Optional[int] = None


class StoryEvent(BaseModel):
    type: Literal["dialogue", "discovery", "combat", "decision"]
    description: str
    participants: List[str] = Field(default_factory=list)
    importance: Literal["low", "normal", "high", "critical"] = Field(default="normal")


class NarrativeFlag(BaseModel):
    name: str
    value: Any


class Observables(BaseModel):
    state_changes: List[StateChange] = Field(default_factory=list)
    story_events: List[StoryEvent] = Field(default_factory=list)
    narrative_flags: List[NarrativeFlag] = Field(default_factory=list)


class Intent(BaseModel):
    goal: str
    conflict: str
    expected_outcome: str

    @field_validator("goal", "conflict", "expected_outcome")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v or len(v.strip()) < 3:
            raise ValueError("字段至少需要3个字符")
        return v.strip()


class ContractMetadata(BaseModel):
    chapter: int
    scene_index: int
    arc: Optional[str] = None
    created_at: Optional[datetime] = Field(default_factory=datetime.now)

    @field_validator("chapter")
    @classmethod
    def validate_chapter(cls, v: int) -> int:
        if v < 1:
            raise ValueError("章号必须大于0")
        return v

    @field_validator("scene_index")
    @classmethod
    def validate_scene_index(cls, v: int) -> int:
        if v < 0:
            raise ValueError("场景序号必须大于等于0")
        return v


class PlanningContract(BaseModel):
    version: str = Field(default="1.0")
    scene_id: str
    intent: Intent
    execution: Execution = Field(default_factory=Execution)
    observables: Observables = Field(default_factory=Observables)
    constraints: List[Constraint] = Field(default_factory=list)
    metadata: ContractMetadata