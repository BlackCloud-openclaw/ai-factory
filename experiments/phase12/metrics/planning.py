from typing import Optional
from ..model import EvaluationContext, MetricResult, MetricState
from .protocol import Metric
from .mixins import AverageAggregateMixin
from ..matching import ExecutionUnitMatcher, RuleExecutionUnitMatcher
from ..config.benchmark import DEFAULT_PASS_THRESHOLD


class PlanningCoverageMetric(AverageAggregateMixin, Metric):
    name = "planning_coverage"
    version = "1.0"

    def __init__(
        self,
        matcher: Optional[ExecutionUnitMatcher] = None,
        pass_threshold: float = DEFAULT_PASS_THRESHOLD,
    ):
        self._matcher = matcher or RuleExecutionUnitMatcher()
        self._pass_threshold = pass_threshold

    async def evaluate(self, ctx: EvaluationContext) -> MetricResult:
        units = ctx.planning_contract.execution.units
        total = len(units)

        if total == 0:
            return MetricResult(
                name=self.name,
                score=1.0,
                state=MetricState.OK,
                raw_value={"covered": 0, "total": 0},
                details={"message": "No units to cover"},
                passed=True,
            )

        covered_unit_ids = []
        missing_unit_ids = []
        for unit in units:
            result = self._matcher.covers(unit, ctx.events)
            if result.matched:
                covered_unit_ids.append(unit.id)
            else:
                missing_unit_ids.append(unit.id)

        covered = len(covered_unit_ids)
        score = covered / total
        passed = score >= self._pass_threshold

        return MetricResult(
            name=self.name,
            score=score,
            state=MetricState.OK,
            raw_value={"covered": covered, "total": total},
            details={
                "covered_unit_ids": covered_unit_ids,
                "missing_unit_ids": missing_unit_ids,
                "total": total,
                "covered": covered,
            },
            passed=passed,
        )