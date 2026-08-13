# src/narrative/adaptive/telemetry.py

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional
from src.narrative.adaptive.provider import StrategyDecisionProvider
from src.narrative.adaptive.model import StrategyDecision

logger = logging.getLogger("telemetry")


class TelemetryDecisionWrapper(StrategyDecisionProvider):
    def __init__(
        self,
        provider: StrategyDecisionProvider,
        novel_id: Optional[str] = None,
        chapter: Optional[int] = None,
        scene: Optional[int] = None,
        rollout_percentage: int = 0,
    ):
        self.provider = provider
        self.novel_id = novel_id
        self.chapter = chapter
        self.scene = scene
        self.rollout_percentage = rollout_percentage

    def decide(self, conflicts, intents) -> StrategyDecision:
        decision = self.provider.decide(conflicts, intents)
        self._log_decision(decision, conflicts, intents)
        return decision

    def _log_decision(self, decision: StrategyDecision, conflicts, intents) -> None:
        if not logger.isEnabledFor(logging.INFO):
            return

        decision_id = str(uuid.uuid4())

        conflict_features = self._extract_conflict_features(conflicts, intents)

        provider_name = "unknown"
        bucket = None
        if hasattr(self.provider, "last_provider"):
            provider_name = self.provider.last_provider or "unknown"
        if hasattr(self.provider, "last_bucket"):
            bucket = self.provider.last_bucket

        record = {
            "decision_id": decision_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "novel_id": self.novel_id,
            "chapter": self.chapter,
            "scene": self.scene,
            "conflict": conflict_features,
            "decision": {
                "strategy": decision.strategy.value,
                "selected_by": decision.selected_by,
                "confidence": decision.confidence,
                "historical_score": decision.historical_score,
            },
            "routing": {
                "provider": provider_name,
                "rollout_percentage": self.rollout_percentage,
                "bucket": bucket,
            },
        }
        logger.info("TELEMETRY_DECISION " + json.dumps(record, default=str))

    def _extract_conflict_features(self, conflicts, intents) -> dict:
        if not conflicts:
            return {"type": "none", "priority_gap": "none"}

        conflict = conflicts[0]
        dimension = conflict.metadata.get("dimension", "unknown")

        conflict_ids = [i.id for i in conflict.intents]
        conflict_intents = [i for i in intents if i.id in conflict_ids]
        priority_gap = "none"
        if len(conflict_intents) >= 2:
            priorities = [i.priority for i in conflict_intents]
            if len(set(priorities)) > 1:
                priority_gap = "high"
            elif len(set(priorities)) == 1:
                priority_gap = "same"

        return {
            "type": dimension,
            "priority_gap": priority_gap,
        }