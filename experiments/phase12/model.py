"""
评测数据模型（不可变）
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence, Any
from enum import Enum
from uuid import UUID, uuid4

from src.writing.planning_contract import PlanningContract
from src.writing.events import NarrativeEvent
from src.writing.snapshot.runtime.models import RuntimeSnapshot


class MetricState(Enum):
    OK = "ok"
    FAILED = "failed"
    PARTIAL = "partial"
    MISSING = "missing"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class EvaluationContext:
    planning_contract: PlanningContract
    scene_text: str
    events: Sequence[NarrativeEvent]

    snapshot_before: Optional[RuntimeSnapshot] = None
    snapshot_after: Optional[RuntimeSnapshot] = None
    revision_result: Optional[dict] = None
    runtime_metrics: Optional[dict] = None

    # 使用 Any 避免循环导入，实际类型为 JudgeContext
    judge_context: Optional[Any] = None

    novel_id: str = ""
    volume: int = 0
    chapter: int = 0
    scene_idx: int = 0

    sample_id: UUID = field(default_factory=uuid4)


@dataclass(frozen=True)
class MetricResult:
    name: str
    score: Optional[float] = None
    state: MetricState = MetricState.OK
    raw_value: Optional[object] = None
    details: dict = field(default_factory=dict)
    passed: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "score": self.score,
            "state": self.state.value,
            "raw_value": self.raw_value,
            "details": self.details,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class BenchmarkResult:
    overall_score: float
    metric_results: Sequence[MetricResult]
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "overall_score": self.overall_score,
            "metric_results": [m.to_dict() for m in self.metric_results],
            "metadata": self.metadata,
        }