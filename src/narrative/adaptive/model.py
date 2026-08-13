# src/narrative/adaptive/model.py

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Optional
from uuid import UUID, uuid4

from src.narrative.conflict import ConflictStrategy


class SelectionMode(StrEnum):
    DETERMINISTIC = "deterministic"
    RULE_BASED = "rule_based"
    ADAPTIVE = "adaptive"


@dataclass(frozen=True)
class StrategyPerformance:
    strategy: ConflictStrategy
    total_uses: int = 0
    total_satisfaction: float = 0.0
    avg_satisfaction: float = 0.0
    total_iterations: int = 0
    avg_iterations: float = 0.0
    last_used: Optional[datetime] = None
    success_rate: float = 0.0

    def update(self, satisfaction: float, iterations: int) -> "StrategyPerformance":
        if not 0.0 <= satisfaction <= 1.0:
            raise ValueError(f"satisfaction must be between 0 and 1, got {satisfaction}")
        if iterations < 0:
            raise ValueError(f"iterations must be >= 0, got {iterations}")

        new_uses = self.total_uses + 1
        new_total_sat = self.total_satisfaction + satisfaction
        new_total_iter = self.total_iterations + iterations
        new_avg_sat = new_total_sat / new_uses
        new_avg_iter = new_total_iter / new_uses
        return StrategyPerformance(
            strategy=self.strategy,
            total_uses=new_uses,
            total_satisfaction=new_total_sat,
            avg_satisfaction=new_avg_sat,
            total_iterations=new_total_iter,
            avg_iterations=new_avg_iter,
            last_used=datetime.now(),
            success_rate=new_avg_sat,
        )


@dataclass(frozen=True)
class StrategyFeedbackEvent:
    conflict_id: UUID
    strategy: ConflictStrategy
    satisfaction_score: float
    iterations: int
    event_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.now)
    resolution_id: Optional[UUID] = None


# ✅ 重新导出 StrategyDecision（从 conflict.model 导入，保持向后兼容）
from src.narrative.conflict.model import StrategyDecision