"""
Benchmark Runner：调度 Metric 执行，不承担聚合策略
"""

from typing import List
from .model import EvaluationContext, BenchmarkResult
from .metrics.registry import MetricRegistry
from .aggregator import ScoreAggregator


class BenchmarkRunner:
    def __init__(
        self,
        registry: MetricRegistry,
        aggregator: ScoreAggregator = None,
    ):
        self._registry = registry
        self._aggregator = aggregator or ScoreAggregator()

    def run(self, contexts: List[EvaluationContext]) -> BenchmarkResult:
        all_aggregated_results = []

        for metric in self._registry.all():
            sample_results = [metric.evaluate(ctx) for ctx in contexts]
            aggregated = metric.aggregate(sample_results)
            all_aggregated_results.append(aggregated)

        overall = self._aggregator.aggregate(all_aggregated_results)

        return BenchmarkResult(
            overall_score=overall,
            metric_results=all_aggregated_results,
            metadata={
                "total_samples": len(contexts),
                "metrics_count": len(self._registry.all()),
            }
        )