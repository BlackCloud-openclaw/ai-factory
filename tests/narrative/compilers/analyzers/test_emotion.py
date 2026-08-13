import pytest

from src.narrative.compilers.analyzers.emotion import EmotionAnalyzer
from tests.narrative.compilers.analyzers.test_dialogue import MockObservation


class TestEmotionAnalyzer:
    def test_returns_intent_when_below_threshold(self):
        obs = MockObservation({"emotion_score": 0.20})
        analyzer = EmotionAnalyzer(min_score=0.4)
        intents = analyzer.analyze(obs, {})

        assert len(intents) == 1
        assert "情绪表达" in intents[0].desired_effect

    def test_returns_empty_when_above_threshold(self):
        obs = MockObservation({"emotion_score": 0.60})
        analyzer = EmotionAnalyzer(min_score=0.4)
        intents = analyzer.analyze(obs, {})

        assert len(intents) == 0