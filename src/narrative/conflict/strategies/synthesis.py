# src/narrative/conflict/strategies/synthesis.py

import logging
from typing import Tuple

from src.narrative.intent.model import NarrativeIntent
from src.narrative.intent.conflict import Conflict
from src.narrative.conflict.model import ConflictResolution, ConflictStrategy
from src.narrative.conflict.protocol import ConflictResolver

logger = logging.getLogger(__name__)


class SynthesisResolver(ConflictResolver):
    """
    合成策略：生成更高层次目标（非二选一）。
    Phase 9.3.2: 使用规则 fallback，不依赖 LLM，保持纯决策层。
    """

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

            rationale = self._generate_synthesis(conflict_intents)

            resolutions.append(
                ConflictResolution(
                    conflict_id=conflict.id,
                    strategy=ConflictStrategy.SYNTHESIS,
                    selected_intent=None,
                    chosen_direction=None,
                    rationale=rationale,
                    affected_intents=tuple(i.id for i in conflict_intents),
                )
            )

        return tuple(resolutions)

    def _generate_synthesis(self, intents: Tuple[NarrativeIntent, ...]) -> str:
        """规则合成，不依赖 LLM"""
        descs = [i.desired_effect for i in intents]
        if len(descs) == 1:
            return f"将「{descs[0]}」作为核心叙事目标推进。"

        joined = " 与 ".join(descs)
        return f"在更高层面统一 '{joined}'，创造超越二选一的叙事价值。"