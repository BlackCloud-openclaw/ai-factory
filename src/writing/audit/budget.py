# src/writing/audit/budget.py
"""
Phase 10.2.4: Budget Analyzer — 预算分析（支持多 Metric，结构化异常）
"""

from types import MappingProxyType
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Optional, Any, Union
from enum import Enum

from ..metrics import MetricName
from .trace import ExecutionTrace


_EMPTY_MAPPING: Mapping[str, float] = MappingProxyType({})


class BudgetSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class BudgetAnomalyKind(Enum):
    HIGH_USAGE = "high_usage"
    LOW_USAGE = "low_usage"
    INVALID_VALUE = "invalid_value"
    NO_DATA = "no_data"
    UNKNOWN_METRIC = "unknown_metric"


@dataclass(frozen=True)
class BudgetAnomaly:
    stage: str
    metric: str
    severity: BudgetSeverity
    kind: BudgetAnomalyKind
    value: Optional[Union[int, float]] = None
    percentage: float = 0.0
    threshold: float = 0.0
    raw_value: Any = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "metric": self.metric,
            "severity": self.severity.value,
            "kind": self.kind.value,
            "value": self.value,
            "percentage": self.percentage,
            "threshold": self.threshold,
            "raw_value": self.raw_value,
            "message": self.message,
        }


@dataclass(frozen=True)
class StageMetricBudget:
    stage: str
    metric: str
    metric_value: Union[int, float]
    percentage: float
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "metric": self.metric,
            "metric_value": self.metric_value,
            "percentage": self.percentage,
            "score": self.score,
        }


@dataclass(frozen=True)
class BudgetReport:
    execution_id: str
    metric: str
    total_metric_value: Union[int, float]
    stages: Sequence[StageMetricBudget] = field(default_factory=tuple)
    anomalies: Sequence[BudgetAnomaly] = field(default_factory=tuple)
    stage_scores: Mapping[str, float] = field(default_factory=lambda: _EMPTY_MAPPING)

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "metric": self.metric,
            "total_metric_value": self.total_metric_value,
            "stages": [s.to_dict() for s in self.stages],
            "anomalies": [a.to_dict() for a in self.anomalies],
            "stage_scores": dict(self.stage_scores),
        }

    def to_markdown(self) -> str:
        lines = [
            "# Budget Analysis Report",
            "",
            f"**Execution ID:** `{self.execution_id}`",
            f"**Metric:** `{self.metric}`",
            f"**Total Value:** {self.total_metric_value:,}",
            "",
            "## Allocation by Stage",
            "",
            "| Stage | Value | Percentage | Score |",
            "|-------|-------|------------|-------|",
        ]
        for s in self.stages:
            lines.append(f"| {s.stage} | {s.metric_value:,} | {s.percentage:.1%} | {s.score:.2f} |")
        lines.append("")
        if self.anomalies:
            lines.append("## Anomalies Detected")
            lines.append("")
            for a in self.anomalies:
                lines.append(f"- [{a.severity.value}] [{a.kind.value}] {a.message}")
        else:
            lines.append("## No anomalies detected.")
        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: dict) -> "BudgetReport":
        return cls(
            execution_id=data.get("execution_id", ""),
            metric=data.get("metric", "tokens"),
            total_metric_value=data.get("total_metric_value", 0),
            stages=(),
            anomalies=(),
            stage_scores=data.get("stage_scores", {}),
        )


class MetricBudgetAnalyzer:
    DEFAULT_HIGH_THRESHOLD = 0.50
    DEFAULT_LOW_THRESHOLD = 0.05

    def __init__(
        self,
        metric: Union[str, MetricName] = MetricName.TOKENS,
        high_threshold: float = DEFAULT_HIGH_THRESHOLD,
        low_threshold: float = DEFAULT_LOW_THRESHOLD,
        check_unknown_metric: bool = False,
    ):
        self._metric = self._normalize_metric(metric)
        self._high_threshold = high_threshold
        self._low_threshold = low_threshold
        self._check_unknown_metric = check_unknown_metric

    @staticmethod
    def _normalize_metric(metric: Union[str, MetricName]) -> str:
        if isinstance(metric, MetricName):
            return metric.value
        if not isinstance(metric, str):
            raise TypeError("metric must be str or MetricName")
        return metric

    def analyze(self, trace: ExecutionTrace) -> BudgetReport:
        known_metrics = {m.value for m in MetricName}
        anomalies: list[BudgetAnomaly] = []

        if self._check_unknown_metric and self._metric not in known_metrics:
            anomalies.append(BudgetAnomaly(
                stage="",
                metric=self._metric,
                severity=BudgetSeverity.INFO,
                kind=BudgetAnomalyKind.UNKNOWN_METRIC,
                value=None,
                percentage=0.0,
                threshold=0.0,
                raw_value=None,
                message=f"Metric '{self._metric}' is not in predefined MetricName list. "
                        f"Known metrics: {sorted(known_metrics)}",
            ))

        stage_metrics: dict[str, Union[int, float]] = {}
        total_metric_value = 0

        for stage in trace.stages:
            raw_value = stage.metrics.get(self._metric)
            if raw_value is None:
                continue
            try:
                if isinstance(raw_value, (int, float)):
                    value = raw_value
                elif isinstance(raw_value, str):
                    if '.' in raw_value:
                        value = float(raw_value)
                    else:
                        value = int(raw_value)
                else:
                    raise ValueError(f"Unsupported metric type: {type(raw_value)}")
            except (ValueError, TypeError):
                anomalies.append(BudgetAnomaly(
                    stage=stage.stage,
                    metric=self._metric,
                    severity=BudgetSeverity.WARNING,
                    kind=BudgetAnomalyKind.INVALID_VALUE,
                    value=None,
                    percentage=0.0,
                    threshold=0.0,
                    raw_value=raw_value,
                    message=f"Invalid metric value: '{raw_value}' (expected numeric)",
                ))
                continue

            stage_metrics[stage.stage] = stage_metrics.get(stage.stage, 0) + value
            total_metric_value += value

        if total_metric_value == 0:
            no_other_anomalies = all(
                a.kind not in (BudgetAnomalyKind.INVALID_VALUE, BudgetAnomalyKind.UNKNOWN_METRIC)
                for a in anomalies
            )
            if no_other_anomalies:
                anomalies.append(BudgetAnomaly(
                    stage="",
                    metric=self._metric,
                    severity=BudgetSeverity.INFO,
                    kind=BudgetAnomalyKind.NO_DATA,
                    value=None,
                    percentage=0.0,
                    threshold=0.0,
                    raw_value=None,
                    message=f"No '{self._metric}' data found in trace.",
                ))
            return BudgetReport(
                execution_id=str(trace.execution_id),
                metric=self._metric,
                total_metric_value=0,
                stages=(),
                anomalies=tuple(anomalies),
                stage_scores=_EMPTY_MAPPING,
            )

        stage_scores: dict[str, float] = {}
        stages_budget: list[StageMetricBudget] = []
        total_stages = len(stage_metrics)
        ideal = 1.0 / total_stages if total_stages > 0 else 0.0

        for stage, value in stage_metrics.items():
            percentage = value / total_metric_value
            if ideal == 0:
                score = 0.0
            else:
                deviation = abs(percentage - ideal) / ideal
                score = min(1.0, deviation)
            stage_scores[stage] = score
            stages_budget.append(StageMetricBudget(
                stage=stage,
                metric=self._metric,
                metric_value=value,
                percentage=percentage,
                score=score,
            ))

        stages_budget.sort(key=lambda x: (-x.metric_value, x.stage))

        for sb in stages_budget:
            if sb.percentage > self._high_threshold:
                anomalies.append(BudgetAnomaly(
                    stage=sb.stage,
                    metric=self._metric,
                    severity=BudgetSeverity.CRITICAL,
                    kind=BudgetAnomalyKind.HIGH_USAGE,
                    value=sb.metric_value,
                    percentage=sb.percentage,
                    threshold=self._high_threshold,
                    raw_value=None,
                    message=(
                        f"Stage '{sb.stage}' consumes {sb.percentage:.1%} of total {self._metric} "
                        f"({sb.metric_value:,}), exceeding threshold {self._high_threshold:.0%}."
                    ),
                ))
            elif sb.percentage < self._low_threshold and sb.percentage > 0:
                anomalies.append(BudgetAnomaly(
                    stage=sb.stage,
                    metric=self._metric,
                    severity=BudgetSeverity.WARNING,
                    kind=BudgetAnomalyKind.LOW_USAGE,
                    value=sb.metric_value,
                    percentage=sb.percentage,
                    threshold=self._low_threshold,
                    raw_value=None,
                    message=(
                        f"Stage '{sb.stage}' consumes only {sb.percentage:.1%} of total {self._metric} "
                        f"({sb.metric_value:,}), below threshold {self._low_threshold:.0%}."
                    ),
                ))

        return BudgetReport(
            execution_id=str(trace.execution_id),
            metric=self._metric,
            total_metric_value=total_metric_value,
            stages=tuple(stages_budget),
            anomalies=tuple(anomalies),
            stage_scores=MappingProxyType(stage_scores),
        )


class BudgetAnalyzer(MetricBudgetAnalyzer):
    pass