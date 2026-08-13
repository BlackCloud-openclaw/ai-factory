# src/narrative/conflict/strategies/priority.py

import logging
from typing import Tuple

from src.narrative.intent.model import NarrativeIntent, IntentPriority
from src.narrative.intent.conflict import Conflict
from src.narrative.conflict.model import ConflictResolution, ConflictStrategy
from src.narrative.conflict.protocol import ConflictResolver

logger = logging.getLogger(__name__)


class PriorityResolver(ConflictResolver):
    """按优先级选择（PRIORITY）"""

    def resolve(
        self,
        conflicts: Tuple[Conflict, ...],
        intents: Tuple[NarrativeIntent, ...],
    ) -> Tuple[ConflictResolution, ...]:
        if not conflicts:
            return ()

        resolutions = []
        for conflict in conflicts:
            conflict_ids = [i.id for i in conflict.intents]
            conflict_intents = [i for i in intents if i.id in conflict_ids]

            if not conflict_intents:
                logger.warning(f"Conflict {conflict.id}: no matching intents, generating ASK")
                resolutions.append(
                    ConflictResolution(
                        conflict_id=conflict.id,
                        strategy=ConflictStrategy.ASK,
                        rationale="无法定位冲突意图，需要外部裁决",
                        affected_intents=tuple(conflict_ids),
                    )
                )
                continue

            sorted_intents = sorted(
                conflict_intents,
                key=lambda i: self._priority_weight(i.priority),
                reverse=True
            )
            chosen = sorted_intents[0]

            resolutions.append(
                ConflictResolution(
                    conflict_id=conflict.id,
                    strategy=ConflictStrategy.PRIORITY,
                    selected_intent=chosen.id,
                    chosen_direction=chosen.dimension.direction,
                    rationale=f"按优先级选择 '{chosen.desired_effect}' (优先级 {chosen.priority.value})",
                    affected_intents=tuple(i.id for i in sorted_intents),
                )
            )

        return tuple(resolutions)

    @staticmethod
    def _priority_weight(priority: IntentPriority) -> int:
        weights = {
            IntentPriority.HIGH: 3,
            IntentPriority.MEDIUM: 2,
            IntentPriority.LOW: 1,
        }
        return weights.get(priority, 0)