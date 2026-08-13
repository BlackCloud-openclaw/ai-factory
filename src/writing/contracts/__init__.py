# src/writing/contracts/__init__.py
from .exceptions import *
from dataclasses import dataclass, field
from typing import Optional
from src.writing.narrative_intent import NarrativeIntent
from src.writing.scene_execution_context import SceneExecutionContext
from src.writing.planning_contract import PlanningContract
from .event_mapping import ContractEventResolver
from .event_matcher import ContractEventMatcher

@dataclass(frozen=True)
class WritingConstraints:
    must_events: list[str] = field(default_factory=list)
    forbidden_events: list[str] = field(default_factory=list)

@dataclass(frozen=True)
class WritingGoal:
    goal: str
    conflict: str
    expected_outcome: str = ""

    def to_prompt(self) -> list[str]:
        lines = []
        if self.goal:
            lines.append(f"🎯 场景目标：{self.goal}")
        if self.conflict:
            lines.append(f"⚔️ 核心冲突：{self.conflict}")
        if self.expected_outcome:
            lines.append(f"🏁 预期结果：{self.expected_outcome}")
        return lines

@dataclass(frozen=True)
class WritingContract:
    scene_context: SceneExecutionContext
    narrative_intent: Optional[NarrativeIntent] = None
    constraints: Optional[WritingConstraints] = None
    writing_goal: Optional[WritingGoal] = None
    execution_contract: Optional[PlanningContract] = None   # ✅ 已添加