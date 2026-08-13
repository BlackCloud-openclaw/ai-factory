# src/narrative/compilers/analyzers/emotion.py

from typing import List

from src.narrative.intent import (
    NarrativeIntent,
    IntentSource,
    IntentPriority,
    IntentDimension,
    IntentDirection,
    BuiltinDimensions,
)
from src.narrative.compilers.interfaces import IntentAnalyzer
from src.runtime.observation.interfaces import ObservationProtocol


class EmotionAnalyzer(IntentAnalyzer):
    def __init__(self, min_score: float = 0.4):
        self._min_score = min_score

    def analyze(
        self,
        observation: ObservationProtocol,
        context: dict,
    ) -> List[NarrativeIntent]:
        emotion_score = observation.get_dimension("emotion_score")

        if emotion_score is None or emotion_score >= self._min_score:
            return []

        return [
            NarrativeIntent(
                source=IntentSource.SYSTEM,
                dimension=IntentDimension(
                    id=BuiltinDimensions.EMOTION,
                    direction=IntentDirection.INCREASE,
                ),
                desired_effect="增强关键时刻的情绪表达，让读者能感受到人物的内心变化",
                preserve=("保持剧情推进", "人物性格"),
                avoid=("过度煽情", "破坏节奏"),
                priority=IntentPriority.MEDIUM,
                rationale=(
                    f"情绪表达评分 {emotion_score:.2f}，"
                    f"低于建议值 {self._min_score:.2f}"
                ),
            )
        ]