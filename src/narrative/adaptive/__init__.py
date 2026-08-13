# src/narrative/adaptive/__init__.py

from .model import (
    StrategyPerformance,
    StrategyFeedbackEvent,
    SelectionMode,
)
from .repository import PerformanceRepository, InMemoryRepository
from .tracker import StrategyPerformanceTracker
from .feedback import StrategyFeedbackCollector
from .adaptive_selector import AdaptiveSelector
from .factory import (
    create_default_adaptive_components,
    create_adaptive_resolver,
    create_deterministic_resolver,
)
from .errors import AdaptiveError, InsufficientDataError

# ✅ 从 conflict.selectors 重新导出 RuleSelector
from src.narrative.conflict.selectors import RuleSelector
# 从 conflict 重新导出 StrategyDecision, StrategyDecisionProvider
from src.narrative.conflict.model import StrategyDecision
from src.narrative.conflict.provider import StrategyDecisionProvider

from .router import StrategyProviderRouter
from .telemetry import TelemetryDecisionWrapper
from .factory import create_adaptive_resolver_with_rollout

__all__ = [
    "StrategyPerformance",
    "StrategyFeedbackEvent",
    "StrategyDecision",
    "SelectionMode",
    "PerformanceRepository",
    "InMemoryRepository",
    "StrategyPerformanceTracker",
    "StrategyFeedbackCollector",
    "RuleSelector",               # ✅ 确保这里包含
    "AdaptiveSelector",
    "StrategyDecisionProvider",
    "create_default_adaptive_components",
    "create_adaptive_resolver",
    "create_deterministic_resolver",
    "AdaptiveError",
    "InsufficientDataError",
]