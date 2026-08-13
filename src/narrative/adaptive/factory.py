# src/narrative/adaptive/factory.py

from typing import Optional

from src.narrative.adaptive.adaptive_selector import AdaptiveSelector
from src.narrative.conflict.selectors import RuleSelector  # ✅ 修正导入路径
from src.narrative.adaptive.router import StrategyProviderRouter
from src.narrative.adaptive.telemetry import TelemetryDecisionWrapper
from src.narrative.adaptive.tracker import StrategyPerformanceTracker
from src.narrative.adaptive.repository import InMemoryRepository, PerformanceRepository
from src.narrative.adaptive.model import SelectionMode
from src.narrative.conflict.composite import CompositeResolver
from src.narrative.conflict.protocol import ConflictResolver
from src.narrative.adaptive.feedback import StrategyFeedbackCollector  # ✅ 新增导入


# === 原有工厂函数（保持兼容）===

def create_default_adaptive_components(
    mode: SelectionMode = SelectionMode.ADAPTIVE,
    repository: Optional[PerformanceRepository] = None,
    min_records: int = 5,
    confidence_threshold: float = 0.05,
) -> dict:
    """创建默认的自适应组件（Tracker, Selector, Collector, Resolver）"""
    repo = repository or InMemoryRepository()
    tracker = StrategyPerformanceTracker(repo)

    if mode == SelectionMode.DETERMINISTIC:
        selector = RuleSelector()
    else:
        selector = AdaptiveSelector(
            tracker=tracker,
            mode=mode,
            min_records_for_adaptive=min_records,
            confidence_threshold=confidence_threshold,
        )

    collector = StrategyFeedbackCollector(tracker)
    resolver = CompositeResolver(provider=selector)

    return {
        "tracker": tracker,
        "selector": selector,
        "collector": collector,
        "resolver": resolver,
    }


def create_adaptive_resolver(
    mode: SelectionMode = SelectionMode.ADAPTIVE,
    repository: Optional[PerformanceRepository] = None,
    min_records: int = 5,
    confidence_threshold: float = 0.05,
) -> ConflictResolver:
    return create_default_adaptive_components(
        mode=mode,
        repository=repository,
        min_records=min_records,
        confidence_threshold=confidence_threshold,
    )["resolver"]


def create_deterministic_resolver() -> ConflictResolver:
    return create_adaptive_resolver(mode=SelectionMode.DETERMINISTIC)


# 别名（向后兼容）
create_default_resolver = create_adaptive_resolver


# === 新增：带灰度路由的工厂函数 ===

def create_adaptive_resolver_with_rollout(
    mode: SelectionMode = SelectionMode.ADAPTIVE,
    rollout_percentage: int = 0,
    repository: Optional[PerformanceRepository] = None,
    min_records: int = 5,
    confidence_threshold: float = 0.05,
    enable_telemetry: bool = True,
    novel_id: Optional[str] = None,
    chapter: Optional[int] = None,
    scene: Optional[int] = None,
) -> ConflictResolver:
    repo = repository or InMemoryRepository()
    tracker = StrategyPerformanceTracker(repo)

    adaptive = AdaptiveSelector(
        tracker=tracker,
        mode=mode,
        min_records_for_adaptive=min_records,
        confidence_threshold=confidence_threshold,
    )
    rule = RuleSelector()

    router = StrategyProviderRouter(
        adaptive_provider=adaptive,
        rule_provider=rule,
        rollout_percentage=rollout_percentage,
    )

    provider = router
    if enable_telemetry:
        provider = TelemetryDecisionWrapper(
            provider=router,
            novel_id=novel_id,
            chapter=chapter,
            scene=scene,
            rollout_percentage=rollout_percentage,
        )

    return CompositeResolver(provider=provider)