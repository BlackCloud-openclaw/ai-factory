from typing import Sequence
from ..model import MetricResult, MetricState


class AverageAggregateMixin:
    def aggregate(self, results: Sequence[MetricResult]) -> MetricResult:
        if not results:
            return MetricResult(
                name=self.name,
                score=None,
                state=MetricState.SKIPPED,
                details={"message": "No samples"},
                passed=False,
            )

        valid_results = [r for r in results if r.score is not None]
        missing_count = len(results) - len(valid_results)

        if not valid_results:
            return MetricResult(
                name=self.name,
                score=None,
                state=MetricState.MISSING,
                details={
                    "sample_count": len(results),
                    "valid_count": 0,
                    "missing_count": missing_count,
                    "message": "All samples missing data",
                },
                passed=False,
            )

        avg_score = sum(r.score for r in valid_results) / len(valid_results)
        passed_count = sum(1 for r in valid_results if r.passed)
        valid_count = len(valid_results)

        return MetricResult(
            name=self.name,
            score=avg_score,
            state=MetricState.OK,
            raw_value=avg_score,
            details={
                "sample_count": len(results),
                "valid_count": valid_count,
                "missing_count": missing_count,
                "passed_count": passed_count,
                "failed_count": valid_count - passed_count,
                "average_score": avg_score,
            },
            passed=passed_count == valid_count
        )


class WorstCaseAggregateMixin:
    def aggregate(self, results: Sequence[MetricResult]) -> MetricResult:
        if not results:
            return MetricResult(
                name=self.name,
                score=None,
                state=MetricState.SKIPPED,
                details={"message": "No samples"},
                passed=False,
            )

        valid_results = [r for r in results if r.score is not None]
        missing_count = len(results) - len(valid_results)

        if not valid_results:
            return MetricResult(
                name=self.name,
                score=None,
                state=MetricState.MISSING,
                details={
                    "sample_count": len(results),
                    "valid_count": 0,
                    "missing_count": missing_count,
                    "message": "All samples missing data",
                },
                passed=False,
            )

        worst_score = min(r.score for r in valid_results)
        passed_count = sum(1 for r in valid_results if r.passed)
        valid_count = len(valid_results)

        return MetricResult(
            name=self.name,
            score=worst_score,
            state=MetricState.OK,
            raw_value=worst_score,
            details={
                "sample_count": len(results),
                "valid_count": valid_count,
                "missing_count": missing_count,
                "passed_count": passed_count,
                "failed_count": valid_count - passed_count,
                "worst_score": worst_score,
            },
            passed=passed_count == valid_count
        )


class PercentileAggregateMixin:
    def __init__(self, percentile: float = 50.0):
        self._percentile = percentile

    def aggregate(self, results: Sequence[MetricResult]) -> MetricResult:
        if not results:
            return MetricResult(
                name=self.name,
                score=None,
                state=MetricState.SKIPPED,
                details={"message": "No samples"},
                passed=False,
            )

        valid_results = [r for r in results if r.score is not None]
        missing_count = len(results) - len(valid_results)

        if not valid_results:
            return MetricResult(
                name=self.name,
                score=None,
                state=MetricState.MISSING,
                details={
                    "sample_count": len(results),
                    "valid_count": 0,
                    "missing_count": missing_count,
                    "message": "All samples missing data",
                },
                passed=False,
            )

        scores = sorted(r.score for r in valid_results)
        import math
        idx = int(math.ceil((self._percentile / 100.0) * len(scores))) - 1
        idx = max(0, min(idx, len(scores) - 1))
        percentile_score = scores[idx]
        passed_count = sum(1 for r in valid_results if r.passed)
        valid_count = len(valid_results)

        return MetricResult(
            name=self.name,
            score=percentile_score,
            state=MetricState.OK,
            raw_value=percentile_score,
            details={
                "sample_count": len(results),
                "valid_count": valid_count,
                "missing_count": missing_count,
                "passed_count": passed_count,
                "failed_count": valid_count - passed_count,
                "percentile": self._percentile,
                "percentile_score": percentile_score,
            },
            passed=passed_count == valid_count
        )