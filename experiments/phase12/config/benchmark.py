from typing import List, Dict, Any

# 权重配置（单一来源）
DEFAULT_WEIGHTS: Dict[str, float] = {
    "planning_coverage": 0.15,
    "state_consistency": 0.15,
    "runtime_health": 0.10,
    "revision_pass_rate": 0.10,
    "continuity": 0.15,
    "character": 0.15,
    "dialogue": 0.10,
    "flow": 0.10,
}

DEFAULT_PASS_THRESHOLD = 0.5

GOLDEN_TOLERANCE = 1e-6
JUDGE_GOLDEN_TOLERANCE = 0.05

BENCHMARK_VERSION = "12.2"

# Golden 文件路径
GOLDEN_DETERMINISTIC_PATH = "experiments/phase12/golden/deterministic.json"
GOLDEN_JUDGE_PATH = "experiments/phase12/golden/judge.json"

# LLM Judge 配置
JUDGE_MODEL = "Qwen3-32B-Q5_K_M"
JUDGE_API_BASE = "http://localhost:8082"
JUDGE_MAX_CONCURRENCY = 10
JUDGE_CACHE_TTL = 86400  # 24 小时

# RuntimeHealthMetric 配置
RUNTIME_HEALTH_CONFIG: Dict[str, Any] = {
    "retry_penalty": 0.05,
    "fallback_penalty": 0.2,
    "error_penalty": 0.3,
    "validation_weight": 0.5,
    "max_penalty": 1.0,
}

# 默认 Metric 类列表（使用完整导入路径）
DEFAULT_METRIC_CLASSES: List[str] = [
    "experiments.phase12.metrics.planning.PlanningCoverageMetric",
    "experiments.phase12.metrics.state.StateConsistencyMetric",
    "experiments.phase12.metrics.runtime.RuntimeHealthMetric",
    "experiments.phase12.metrics.revision.RevisionPassRateMetric",
    "experiments.phase12.judge.metric.ContinuityJudgeMetric",
    "experiments.phase12.judge.metric.CharacterJudgeMetric",
    "experiments.phase12.judge.metric.DialogueJudgeMetric",
    "experiments.phase12.judge.metric.FlowJudgeMetric",
]

SUMMARY_KEYS = [
    "sample_count", "passed_count", "failed_count",
    "total", "covered", "matched", "missing_count",
    "retry_count", "fallback_count", "error_count",
    "validation_score", "after_compliance", "total_penalty",
    "confidence", "tokens_used", "elapsed_ms",
]