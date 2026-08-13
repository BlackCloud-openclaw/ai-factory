import asyncio
from typing import List, Optional
from ..model import EvaluationContext, BenchmarkResult, MetricState, MetricResult
from ..metrics.registry import MetricRegistry
from ..aggregator import ScoreAggregator
from ..config.benchmark import BENCHMARK_VERSION


class BenchmarkRunner:
    def __init__(
        self,
        registry: MetricRegistry,
        aggregator: Optional[ScoreAggregator] = None,
    ):
        self._registry = registry
        self._aggregator = aggregator or ScoreAggregator.default()

    async def run(self, contexts: List[EvaluationContext]) -> BenchmarkResult:
        """执行 Benchmark，所有 Metric 并发评估所有样本。"""
        metrics = self._registry.all()
        tasks = []
        task_metadata = []

        for metric in metrics:
            for idx, ctx in enumerate(contexts):
                tasks.append(metric.evaluate(ctx))
                task_metadata.append((metric.name, idx))

        # 统一调度所有任务（LLM 限流由 JudgeClient 内部管理）
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 按 Metric 聚合结果
        metric_results_map = {m.name: [] for m in metrics}
        error_count = 0
        success_count = 0

        for (name, _), result in zip(task_metadata, results):
            if isinstance(result, Exception):
                error_count += 1
                metric_results_map[name].append(MetricResult(
                    name=name,
                    score=None,
                    state=MetricState.ERROR,
                    details={"error": str(result)},
                    passed=False,
                ))
            else:
                success_count += 1
                metric_results_map[name].append(result)

        # 聚合每个 Metric
        aggregated_results = []
        for metric in metrics:
            agg = metric.aggregate(metric_results_map[metric.name])
            aggregated_results.append(agg)

        overall = self._aggregator.aggregate(aggregated_results)

        return BenchmarkResult(
            overall_score=overall,
            metric_results=aggregated_results,
            metadata={
                "benchmark_version": BENCHMARK_VERSION,
                "total_samples": len(contexts),
                "total_tasks": len(tasks),
                "success_count": success_count,
                "error_count": error_count,
                "metrics": [m.name for m in self._registry.all()],
                "metrics_count": len(self._registry.all()),
                "aggregator": self._aggregator.to_dict(),
            }
        )