# src/narrative/compilers/analyzers/dialogue.py

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


class DialogueAnalyzer(IntentAnalyzer):
    def __init__(self, min_ratio: float = 0.20):
        self._min_ratio = min_ratio

    def analyze(
        self,
        observation: ObservationProtocol,
        context: dict,
    ) -> List[NarrativeIntent]:
        dialogue_ratio = observation.get_dimension("dialogue_ratio")

        if dialogue_ratio is None or dialogue_ratio >= self._min_ratio:
            return []

        return [
            NarrativeIntent(
                source=IntentSource.SYSTEM,
                dimension=IntentDimension(
                    id=BuiltinDimensions.DIALOGUE,
                    direction=IntentDirection.INCREASE,
                ),
                desired_effect="增强人物互动，通过自然交流体现角色关系和信息交换",
                preserve=("保持剧情推进", "保持人物状态"),
                avoid=("信息式对白", "纯说明性对话"),
                priority=IntentPriority.MEDIUM,
                rationale=(
                    f"当前对白占比 {dialogue_ratio:.0%}，"
                    f"低于建议值 {self._min_ratio:.0%}"
                ),
            )
        ]