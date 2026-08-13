# src/writing/snapshot/runtime/remote/cache/metrics.py
"""
B4.2: CacheMetrics — 缓存指标汇总
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CacheMetrics:
    """缓存指标快照（不可变）。"""

    hits: int
    misses: int
    evictions: int
    size: int
    maxsize: int
    remote_reads: int = 0
    remote_writes: int = 0

    @property
    def hit_rate(self) -> float:
        """命中率（0-1）。"""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    @property
    def utilization(self) -> float:
        """缓存利用率（0-1）。"""
        if self.maxsize == 0:
            return 0.0
        return self.size / self.maxsize