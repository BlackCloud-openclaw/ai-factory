from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from src.writing.narrative_intent import NarrativeIntent
from src.writing.scene_execution_context import SceneExecutionContext
from src.writing.planning_contract import PlanningContract  # 新增导入

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
    execution_contract: Optional[PlanningContract] = None
    # Phase 15.8 Commit 2: 上一场景结尾（跨场景衔接）
    previous_scene_tail: Optional[str] = None
    scene_spec: Optional[Dict[str, Any]] = None       # ← 新增