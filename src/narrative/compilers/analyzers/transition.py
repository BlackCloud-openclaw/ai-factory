# src/narrative/compilers/analyzers/transition.py

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


class TransitionAnalyzer(IntentAnalyzer):
    def __init__(self, min_score: float = 0.5):
        self._min_score = min_score

    def analyze(
        self,
        observation: ObservationProtocol,
        context: dict,
    ) -> List[NarrativeIntent]:
        transition_score = observation.get_dimension("transition_score")

        if transition_score is None or transition_score >= self._min_score:
            return []

        return [
            NarrativeIntent(
                source=IntentSource.SYSTEM,
                dimension=IntentDimension(
                    id=BuiltinDimensions.TRANSITION,
                    direction=IntentDirection.INCREASE,
                ),
                desired_effect="让场景之间的过渡更加自然，读者不会被突兀的切换打断",
                preserve=("所有剧情事件", "人物状态"),
                avoid=("改变事件顺序", "删除场景"),
                priority=IntentPriority.HIGH,
                rationale=(
                    f"场景衔接评分 {transition_score:.2f}，"
                    f"低于建议值 {self._min_score:.2f}"
                ),
            )
        ]