import pytest

from src.narrative.compilers.analyzers.dialogue import DialogueAnalyzer
from src.runtime.observation.interfaces import ObservationProtocol


class MockObservation(ObservationProtocol):
    def __init__(self, dimensions: dict):
        self._dimensions = dimensions

    def get_dimension(self, name: str) -> float | None:
        return self._dimensions.get(name)

    def get_evidence(self, dimension: str) -> list[str]:
        return []

    def get_all_dimensions(self) -> tuple[str, ...]:
        return tuple(self._dimensions.keys())


class TestDialogueAnalyzer:
    def test_returns_intent_when_below_threshold(self):
        obs = MockObservation({"dialogue_ratio": 0.10})
        analyzer = DialogueAnalyzer(min_ratio=0.20)
        intents = analyzer.analyze(obs, {})

        assert len(intents) == 1
        assert "增强人物互动" in intents[0].desired_effect
        assert intents[0].priority.value == "medium"

    def test_returns_empty_when_above_threshold(self):
        obs = MockObservation({"dialogue_ratio": 0.30})
        analyzer = DialogueAnalyzer(min_ratio=0.20)
        intents = analyzer.analyze(obs, {})

        assert len(intents) == 0

    def test_returns_empty_when_dimension_missing(self):
        obs = MockObservation({})
        analyzer = DialogueAnalyzer()
        intents = analyzer.analyze(obs, {})

        assert len(intents) == 0