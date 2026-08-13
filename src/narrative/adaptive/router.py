# src/narrative/adaptive/router.py

import hashlib
from typing import Optional
from src.narrative.adaptive.provider import StrategyDecisionProvider
from src.narrative.adaptive.model import StrategyDecision


class StrategyProviderRouter(StrategyDecisionProvider):
    """
    策略提供者路由层。
    根据 novel_id 哈希决定使用 Adaptive 还是 Rule。
    """

    def __init__(
        self,
        adaptive_provider: StrategyDecisionProvider,
        rule_provider: StrategyDecisionProvider,
        rollout_percentage: int = 0,
    ):
        self.adaptive_provider = adaptive_provider
        self.rule_provider = rule_provider
        self.set_rollout_percentage(rollout_percentage)
        self.last_provider: Optional[str] = None
        self.last_bucket: Optional[int] = None

    def decide(self, conflicts, intents) -> StrategyDecision:
        novel_id = self._extract_novel_id(conflicts, intents)
        self.last_provider = "rule"
        self.last_bucket = None

        if novel_id:
            self.last_bucket = int(hashlib.md5(novel_id.encode()).hexdigest(), 16) % 100
            if self.last_bucket < self.rollout_percentage:
                self.last_provider = "adaptive"
                return self.adaptive_provider.decide(conflicts, intents)

        return self.rule_provider.decide(conflicts, intents)

    def _extract_novel_id(self, conflicts, intents) -> Optional[str]:
        for intent in intents:
            if hasattr(intent, "metadata") and intent.metadata:
                return intent.metadata.get("novel_id")
        return None

    def set_rollout_percentage(self, percentage: int) -> None:
        if not 0 <= percentage <= 100:
            raise ValueError("rollout percentage must be between 0 and 100")
        self.rollout_percentage = percentage