from typing import Optional
from ..model import EvaluationContext, MetricResult, MetricState
from .protocol import Metric
from .mixins import AverageAggregateMixin
from ..matching import RuleStateMatcher, StateMatcher
from ..builder import StateExpectationBuilder
from ..config.benchmark import DEFAULT_PASS_THRESHOLD


class StateConsistencyMetric(AverageAggregateMixin, Metric):
    name = "state_consistency"
    version = "1.0"

    def __init__(
        self,
        matcher: Optional[StateMatcher] = None,
        builder: Optional[StateExpectationBuilder] = None,
        pass_threshold: float = DEFAULT_PASS_THRESHOLD,
    ):
        self._matcher = matcher or RuleStateMatcher()
        self._builder = builder or StateExpectationBuilder()
        self._pass_threshold = pass_threshold

    async def evaluate(self, ctx: EvaluationContext) -> MetricResult:
        expected = self._builder.build(ctx.planning_contract)
        if not expected:
            return MetricResult(
                name=self.name,
                score=1.0,
                state=MetricState.OK,
                raw_value={"matched": 0, "total": 0},
                details={"message": "No state changes expected"},
                passed=True,
            )

        results = self._matcher.compare(expected, ctx.snapshot_before, ctx.snapshot_after)
        total = len(results)
        matched = sum(1 for r in results if r.matched)
        score = matched / total if total > 0 else 1.0
        passed = score >= self._pass_threshold

        return MetricResult(
            name=self.name,
            score=score,
            state=MetricState.OK,
            raw_value={"matched": matched, "total": total},
            details={
                "matched_fields": [r.field for r in results if r.matched],
                "missing_fields": [r.field for r in results if not r.matched],
                "total": total,
                "matched": matched,
                "field_results": [
                    {
                        "field": r.field,
                        "expectation_id": r.expectation_id,
                        "matched": r.matched,
                        "strategy": r.strategy,
                        "severity": r.severity.value,
                        "expected": r.expected,
                        "actual": r.actual,
                    }
                    for r in results
                ]
            },
            passed=passed,
        )