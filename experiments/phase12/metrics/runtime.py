from typing import Dict, Any, Optional

from ..model import EvaluationContext, MetricResult, MetricState
from .protocol import Metric
from .mixins import AverageAggregateMixin
from ..config.benchmark import RUNTIME_HEALTH_CONFIG, DEFAULT_PASS_THRESHOLD


class RuntimeHealthMetric(AverageAggregateMixin, Metric):
    name = "runtime_health"
    version = "1.0"

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        pass_threshold: Optional[float] = None,
    ):
        self._config = config or RUNTIME_HEALTH_CONFIG
        self._pass_threshold = pass_threshold or DEFAULT_PASS_THRESHOLD

    async def evaluate(self, ctx: EvaluationContext) -> MetricResult:
        metrics = ctx.runtime_metrics or {}

        if not metrics:
            return MetricResult(
                name=self.name,
                score=None,
                state=MetricState.MISSING,
                details={"message": "No runtime metrics available"},
                passed=False,
            )

        retry_count = metrics.get("retry_count", 0)
        fallback_count = metrics.get("fallback_count", 0)
        error_count = metrics.get("error_count", 0)
        validation_score = metrics.get("validation_score", 1.0)

        try:
            validation_score = max(0.0, min(1.0, float(validation_score)))
        except (TypeError, ValueError):
            validation_score = 0.5

        retry_penalty = min(retry_count * self._config["retry_penalty"], 0.5)
        fallback_penalty = min(fallback_count * self._config["fallback_penalty"], 0.5)
        error_penalty = min(error_count * self._config["error_penalty"], 0.5)
        validation_penalty = (1.0 - validation_score) * self._config["validation_weight"]
        validation_penalty = max(0.0, validation_penalty)

        total_penalty = min(
            retry_penalty + fallback_penalty + error_penalty + validation_penalty,
            self._config["max_penalty"]
        )
        score = max(0.0, 1.0 - total_penalty)

        passed = score >= self._pass_threshold

        return MetricResult(
            name=self.name,
            score=score,
            state=MetricState.OK,
            raw_value={
                "retry_count": retry_count,
                "fallback_count": fallback_count,
                "error_count": error_count,
                "validation_score": validation_score,
            },
            details={
                "retry_count": retry_count,
                "fallback_count": fallback_count,
                "error_count": error_count,
                "validation_score": validation_score,
                "retry_penalty": retry_penalty,
                "fallback_penalty": fallback_penalty,
                "error_penalty": error_penalty,
                "validation_penalty": validation_penalty,
                "total_penalty": total_penalty,
            },
            passed=passed,
        )