# src/writing/audit/metric_names.py
"""
Phase 10.2.4: 预定义 Metric 名称（共享于 Runtime 与 Audit）
"""

from enum import StrEnum


class MetricName(StrEnum):
    """预定义的 Metric 名称。"""
    TOKENS = "tokens"
    LATENCY_MS = "latency_ms"
    COST_USD = "cost_usd"

    # 可扩展，但建议先在 Runtime 和 Audit 间达成一致