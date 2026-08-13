# src/narrative/conflict/strategies/balance.py

import logging
from typing import Tuple

from src.narrative.intent.model import NarrativeIntent
from src.narrative.intent.conflict import Conflict
from src.narrative.conflict.model import ConflictResolution, ConflictStrategy
from src.narrative.conflict.protocol import ConflictResolver

logger = logging.getLogger(__name__)


class BalanceResolver(ConflictResolver):
    """平衡策略：保留双方核心价值，折中实现。"""

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

            desc_a = conflict_intents[0].desired_effect
            desc_b = conflict_intents[1].desired_effect if len(conflict_intents) > 1 else ""

            # 获取 dimension：优先从 conflict 属性，其次 metadata
            dimension = getattr(conflict, "dimension", None)
            if not dimension:
                dimension = conflict.metadata.get("dimension", "未知维度")

            rationale = (
                f"平衡双方目标：保留「{desc_a}」的核心价值，同时兼顾「{desc_b}」的合理性。"
                f"在 {dimension} 维度上寻求折中表达。"
            )

            resolutions.append(
                ConflictResolution(
                    conflict_id=conflict.id,
                    strategy=ConflictStrategy.BALANCE,
                    selected_intent=None,
                    chosen_direction=None,
                    rationale=rationale,
                    affected_intents=tuple(i.id for i in conflict_intents),
                )
            )

        return tuple(resolutions)