# src/narrative/adaptive/adaptive_selector.py

from typing import Tuple, List, Optional

from src.narrative.intent.model import NarrativeIntent
from src.narrative.intent.conflict import Conflict
from src.narrative.conflict import ConflictStrategy
from src.narrative.conflict.provider import StrategyDecisionProvider
from src.narrative.conflict.model import StrategyDecision
from src.narrative.conflict.selectors import RuleSelector  # ✅ 从 conflict 导入
from src.narrative.adaptive.model import SelectionMode
from src.narrative.adaptive.tracker import StrategyPerformanceTracker


class AdaptiveSelector(StrategyDecisionProvider):
    def __init__(
        self,
        tracker: StrategyPerformanceTracker,
        mode: SelectionMode = SelectionMode.ADAPTIVE,
        min_records_for_adaptive: int = 5,
        confidence_threshold: float = 0.05,
    ):
        self._tracker = tracker
        self._rule_selector = RuleSelector()  # ✅ 使用冲突包的 RuleSelector
        self._mode = mode
        self._min_records = min_records_for_adaptive
        self._confidence_threshold = confidence_threshold


    def decide(
        self,
        conflicts: Tuple[Conflict, ...],
        intents: Tuple[NarrativeIntent, ...],
    ) -> StrategyDecision:
        if self._mode == SelectionMode.DETERMINISTIC:
            return self._rule_selector.decide(conflicts, intents)

        rule_decision = self._rule_selector.decide(conflicts, intents)

        total_events = self._tracker.get_event_count()
        if total_events < self._min_records:
            return StrategyDecision(
                strategy=rule_decision.strategy,
                confidence=0.5,
                reason=f"Insufficient data ({total_events} events < {self._min_records})",
                selected_by="fallback_insufficient_data",
                historical_score=None,
            )

        eligible_strategies = self._get_eligible_strategies(conflicts, intents)
        best_eligible = self._select_best_among_eligible(eligible_strategies)

        if best_eligible is None:
            return rule_decision

        best_perf = self._tracker.get_performance(best_eligible)
        rule_perf = self._tracker.get_performance(rule_decision.strategy)

        if best_perf is None:
            return rule_decision

        rule_score = rule_perf.avg_satisfaction if rule_perf else 0.0
        if best_perf.avg_satisfaction > rule_score + self._confidence_threshold:
            return StrategyDecision(
                strategy=best_eligible,
                confidence=best_perf.avg_satisfaction,
                reason=(
                    f"Adaptive: {best_eligible.value} outperforms "
                    f"{rule_decision.strategy.value} "
                    f"({best_perf.avg_satisfaction:.2f} > {rule_score:.2f})"
                ),
                historical_score=best_perf.avg_satisfaction,
                selected_by="adaptive",
            )

        return StrategyDecision(
            strategy=rule_decision.strategy,
            confidence=rule_perf.avg_satisfaction if rule_perf else 0.5,
            reason=f"Rule-based with history: {rule_decision.strategy.value}",
            historical_score=rule_perf.avg_satisfaction if rule_perf else None,
            selected_by="rule_with_history",
        )

    def _get_eligible_strategies(self, conflicts, intents) -> List[ConflictStrategy]:
        features = self._rule_selector._analyze_features(conflicts, intents)
        eligible = []

        if features.get("has_priority_diff") or features.get("is_absolute_conflict"):
            eligible.append(ConflictStrategy.PRIORITY)

        eligible.append(ConflictStrategy.BALANCE)

        if features.get("synthesis_hint"):
            eligible.append(ConflictStrategy.SYNTHESIS)

        if not eligible:
            eligible.append(ConflictStrategy.BALANCE)

        return eligible

    def _select_best_among_eligible(
        self,
        eligible: List[ConflictStrategy],
    ) -> Optional[ConflictStrategy]:
        best_strategy = None
        best_score = -1.0

        for strategy in eligible:
            perf = self._tracker.get_performance(strategy)
            if perf and perf.avg_satisfaction > best_score:
                best_score = perf.avg_satisfaction
                best_strategy = strategy

        return best_strategy

    @property
    def mode(self) -> SelectionMode:
        return self._mode

    @mode.setter
    def mode(self, value: SelectionMode) -> None:
        self._mode = value