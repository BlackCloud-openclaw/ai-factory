# src/writing/audit/priority.py
"""
Phase 10.2.5: Priority Engine — 优化优先级计算（纯评分引擎，策略可注入）
"""

from enum import Enum
from types import MappingProxyType
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Optional, Any

from .preservation import PreservationReport
from .attribution import AttributionReport, AttributionType
from .budget import BudgetReport
from ..stage_names import StageName


_EMPTY_MAPPING = MappingProxyType({})


class PriorityLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class PriorityFactor:
    name: str
    weight: float
    score: float

    @property
    def contribution(self) -> float:
        return self.score * self.weight


@dataclass(frozen=True)
class OptimizationTarget:
    field_name: str
    lost_stage: str
    current_retention: float
    stage_score: float
    severity: PriorityLevel
    priority_score: float
    factors: Sequence[PriorityFactor] = field(default_factory=tuple)
    attribution_type: Optional[AttributionType] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "lost_stage": self.lost_stage,
            "current_retention": self.current_retention,
            "stage_score": self.stage_score,
            "severity": self.severity.value,
            "priority_score": self.priority_score,
            "factors": [
                {
                    "name": f.name,
                    "score": f.score,
                    "weight": f.weight,
                    "contribution": f.contribution,
                }
                for f in self.factors
            ],
            "attribution_type": self.attribution_type.value if self.attribution_type else None,
        }


@dataclass(frozen=True)
class PriorityReport:
    execution_id: str
    targets: Sequence[OptimizationTarget] = field(default_factory=tuple)

    @property
    def total_targets(self) -> int:
        return len(self.targets)

    @property
    def top_critical(self) -> Optional[OptimizationTarget]:
        for t in self.targets:
            if t.severity == PriorityLevel.CRITICAL:
                return t
        return None

    def get_by_severity(self, severity: PriorityLevel) -> Sequence[OptimizationTarget]:
        return tuple(t for t in self.targets if t.severity == severity)

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "total_targets": self.total_targets,
            "targets": [t.to_dict() for t in self.targets],
            "top_critical": self.top_critical.to_dict() if self.top_critical else None,
        }

    def to_markdown(self) -> str:
        lines = [
            "# Priority Report",
            "",
            f"**Execution ID:** `{self.execution_id}`",
            f"**Total Targets:** {self.total_targets}",
            "",
            "## Priority Summary",
            "",
            "| Priority | Field | Stage | Retention | Stage Score | Score |",
            "|----------|-------|-------|-----------|-------------|-------|",
        ]
        for t in self.targets[:10]:
            lines.append(
                f"| {t.severity.value} | {t.field_name} | {t.lost_stage} | "
                f"{t.current_retention:.1%} | {t.stage_score:.2f} | {t.priority_score:.1f} |"
            )
        lines.append("")

        if self.targets:
            lines.append("## Top Recommendation")
            top = self.targets[0]
            lines.append("")
            lines.append(f"### Field: `{top.field_name}` at `{top.lost_stage}`")
            lines.append(f"- Priority Score: {top.priority_score:.1f}")
            lines.append(f"- Current Retention: {top.current_retention:.1%}")
            lines.append(f"- Stage Score: {top.stage_score:.2f}")
            lines.append(f"- Severity: {top.severity.value}")
            if top.attribution_type:
                lines.append(f"- Attribution: {top.attribution_type.value}")
            lines.append("")
            lines.append("#### Factors")
            for f in top.factors:
                lines.append(f"- {f.name}: score={f.score:.2f}, weight={f.weight:.2f}, contribution={f.contribution:.2f}")

        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: dict) -> "PriorityReport":
        return cls(
            execution_id=data.get("execution_id", ""),
            targets=(),
        )


@dataclass(frozen=True)
class PriorityPolicy:
    retention_weight: float = 0.4
    stage_score_weight: float = 0.3
    stage_position_weight: float = 0.2
    attribution_weight: float = 0.1

    stage_position_weights: Mapping[StageName, float] = field(
        default_factory=lambda: _EMPTY_MAPPING
    )
    attribution_weights: Mapping[AttributionType, float] = field(
        default_factory=lambda: _EMPTY_MAPPING
    )
    severity_thresholds: Mapping[PriorityLevel, float] = field(
        default_factory=lambda: _EMPTY_MAPPING
    )

    @classmethod
    def default(cls) -> "PriorityPolicy":
        return cls(
            retention_weight=0.4,
            stage_score_weight=0.3,
            stage_position_weight=0.2,
            attribution_weight=0.1,
            stage_position_weights=MappingProxyType({
                StageName.PLANNING: 1.0,
                StageName.OBSERVATION: 0.9,
                StageName.IR: 0.8,
                StageName.PROMPT: 0.6,
                StageName.DRAFT: 0.4,
                StageName.COVERAGE: 0.2,
            }),
            attribution_weights=MappingProxyType({
                AttributionType.TRANSFORM_LOST: 1.0,
                AttributionType.INPUT_LOST: 0.8,
                AttributionType.UNKNOWN: 0.5,
            }),
            severity_thresholds=MappingProxyType({
                PriorityLevel.CRITICAL: 80.0,
                PriorityLevel.HIGH: 60.0,
                PriorityLevel.MEDIUM: 35.0,
            }),
        )

    def get_stage_position_score(self, stage: StageName) -> float:
        return self.stage_position_weights.get(stage, 0.5)

    def get_stage_position_score_by_name(self, stage: str) -> float:
        stage_enum = StageName.safe_parse(stage)
        if stage_enum is None:
            return 0.5
        return self.get_stage_position_score(stage_enum)

    def get_attribution_score(self, attr_type: Optional[AttributionType]) -> float:
        if attr_type is None:
            return 0.5
        return self.attribution_weights.get(attr_type, 0.5)

    def determine_severity(self, priority_score: float) -> PriorityLevel:
        thresholds = self.severity_thresholds
        if priority_score >= thresholds.get(PriorityLevel.CRITICAL, 80.0):
            return PriorityLevel.CRITICAL
        elif priority_score >= thresholds.get(PriorityLevel.HIGH, 60.0):
            return PriorityLevel.HIGH
        elif priority_score >= thresholds.get(PriorityLevel.MEDIUM, 35.0):
            return PriorityLevel.MEDIUM
        else:
            return PriorityLevel.LOW


class PriorityEngine:
    def __init__(self, policy: Optional[PriorityPolicy] = None):
        self._policy = policy or PriorityPolicy.default()

    def analyze(
        self,
        execution_id: str,
        preservation_report: PreservationReport,
        attribution_report: AttributionReport,
        budget_report: BudgetReport,
    ) -> PriorityReport:
        targets: list[OptimizationTarget] = []

        stage_scores = budget_report.stage_scores

        for field_name, field_pres in preservation_report.fields.items():
            if field_pres.is_fully_preserved:
                continue

            attr = attribution_report.attributions.get(field_name)
            if attr is None:
                continue

            lost_stage_str = attr.lost_stage
            position_score = self._policy.get_stage_position_score_by_name(lost_stage_str)
            attr_type = attr.attribution_type

            retention_score = 1.0 - field_pres.end_retention_rate
            stage_score = stage_scores.get(lost_stage_str, 0.0)
            attr_score = self._policy.get_attribution_score(attr_type)

            raw_score = (
                self._policy.retention_weight * retention_score +
                self._policy.stage_score_weight * stage_score +
                self._policy.stage_position_weight * position_score +
                self._policy.attribution_weight * attr_score
            )
            priority_score = min(100.0, raw_score * 100.0)
            severity = self._policy.determine_severity(priority_score)

            factors = [
                PriorityFactor("Retention Loss", self._policy.retention_weight, retention_score),
                PriorityFactor("Stage Score", self._policy.stage_score_weight, stage_score),
                PriorityFactor("Stage Position", self._policy.stage_position_weight, position_score),
                PriorityFactor("Attribution Type", self._policy.attribution_weight, attr_score),
            ]

            target = OptimizationTarget(
                field_name=field_name,
                lost_stage=lost_stage_str,
                current_retention=field_pres.end_retention_rate,
                stage_score=stage_score,
                severity=severity,
                priority_score=priority_score,
                factors=tuple(factors),
                attribution_type=attr_type,
            )
            targets.append(target)

        targets.sort(key=lambda x: (-x.priority_score, x.field_name, x.lost_stage))

        return PriorityReport(
            execution_id=execution_id,
            targets=tuple(targets),
        )