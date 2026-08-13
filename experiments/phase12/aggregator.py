from typing import Sequence, Literal, Dict, Optional
from .model import MetricResult
from .config.benchmark import DEFAULT_WEIGHTS


AggregationMethod = Literal["arithmetic_mean", "weighted_mean", "geometric_mean", "worst_case"]


class ScoreAggregator:
    def __init__(
        self,
        method: AggregationMethod = "weighted_mean",
        weights: Optional[Dict[str, float]] = None,
    ):
        self._method = method
        self._weights = weights or DEFAULT_WEIGHTS.copy()

    @property
    def weights(self) -> Dict[str, float]:
        return self._weights.copy()

    @classmethod
    def default(cls) -> "ScoreAggregator":
        return cls(method="weighted_mean", weights=DEFAULT_WEIGHTS.copy())

    def aggregate(self, results: Sequence[MetricResult]) -> float:
        if not results:
            return 0.0

        if self._method == "arithmetic_mean":
            return self._arithmetic_mean(results)
        elif self._method == "weighted_mean":
            return self._weighted_mean(results)
        elif self._method == "geometric_mean":
            return self._geometric_mean(results)
        elif self._method == "worst_case":
            return self._worst_case(results)
        else:
            raise ValueError(f"Unknown aggregation method: {self._method}")

    def _arithmetic_mean(self, results: Sequence[MetricResult]) -> float:
        return sum(r.score for r in results if r.score is not None) / len(results)

    def _weighted_mean(self, results: Sequence[MetricResult]) -> float:
        total_weight = 0.0
        weighted_sum = 0.0
        for r in results:
            if r.score is None:
                continue
            w = self._weights.get(r.name, 1.0)
            weighted_sum += r.score * w
            total_weight += w
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _geometric_mean(self, results: Sequence[MetricResult]) -> float:
        import math
        product = 1.0
        count = 0
        for r in results:
            if r.score is not None and r.score > 0:
                product *= r.score
                count += 1
        if count == 0:
            return 0.0
        return math.pow(product, 1.0 / count)

    def _worst_case(self, results: Sequence[MetricResult]) -> float:
        scores = [r.score for r in results if r.score is not None]
        return min(scores) if scores else 0.0

    def to_dict(self) -> Dict:
        return {"method": self._method, "weights": self._weights.copy()}