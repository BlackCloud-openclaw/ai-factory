import pytest

from src.narrative.compilers.analyzers.transition import TransitionAnalyzer
from tests.narrative.compilers.analyzers.test_dialogue import MockObservation


class TestTransitionAnalyzer:
    def test_returns_intent_when_below_threshold(self):
        obs = MockObservation({"transition_score": 0.30})
        analyzer = TransitionAnalyzer(min_score=0.5)
        intents = analyzer.analyze(obs, {})

        assert len(intents) == 1
        assert "场景之间的过渡" in intents[0].desired_effect
        assert intents[0].priority.value == "high"

    def test_returns_empty_when_above_threshold(self):
        obs = MockObservation({"transition_score": 0.70})
        analyzer = TransitionAnalyzer(min_score=0.5)
        intents = analyzer.analyze(obs, {})

        assert len(intents) == 0