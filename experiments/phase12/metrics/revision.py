from typing import Optional

from ..model import EvaluationContext, MetricResult, MetricState
from .protocol import Metric
from .mixins import AverageAggregateMixin
from ..config.benchmark import DEFAULT_PASS_THRESHOLD


class RevisionPassRateMetric(AverageAggregateMixin, Metric):
    name = "revision_pass_rate"
    version = "1.0"

    def __init__(self, pass_threshold: Optional[float] = None):
        self._pass_threshold = pass_threshold or DEFAULT_PASS_THRESHOLD

    async def evaluate(self, ctx: EvaluationContext) -> MetricResult:
        revision = ctx.revision_result or {}
        revision_metrics = revision.get("metrics", {})
        after_compliance = revision_metrics.get("after_compliance")

        if after_compliance is None:
            runtime = ctx.runtime_metrics or {}
            after_compliance = runtime.get("after_compliance") or runtime.get("compliance")

        if after_compliance is None:
            return MetricResult(
                name=self.name,
                score=None,
                state=MetricState.MISSING,
                details={"message": "No revision data available"},
                passed=False,
            )

        try:
            score = min(1.0, max(0.0, float(after_compliance)))
        except (TypeError, ValueError):
            return MetricResult(
                name=self.name,
                score=None,
                state=MetricState.ERROR,
                details={
                    "message": f"Invalid after_compliance value: {after_compliance}",
                    "raw_value": after_compliance,
                },
                passed=False,
            )

        passed = score >= self._pass_threshold

        return MetricResult(
            name=self.name,
            score=score,
            state=MetricState.OK,
            raw_value=after_compliance,
            details={
                "after_compliance": after_compliance,
                "before_compliance": revision_metrics.get("before_compliance"),
                "delta": revision_metrics.get("delta"),
            },
            passed=passed,
        )