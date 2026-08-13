# tests/narrative/compilers/test_intent_compiler.py

import pytest

from src.narrative.intent import (
    NarrativeIntent,
    IntentSource,
    IntentPriority,
    IntentDimension,
    IntentDirection,
    BuiltinDimensions,
)
from src.narrative.compilers.intent_compiler import IntentCompiler
from src.narrative.compilers.interfaces import IntentAnalyzer
from tests.narrative.compilers.analyzers.test_dialogue import MockObservation


class MockAnalyzer(IntentAnalyzer):
    def __init__(self, return_intents: bool = True):
        self._return_intents = return_intents

    def analyze(self, observation, context):
        if self._return_intents:
            return [
                NarrativeIntent(
                    source=IntentSource.SYSTEM,
                    dimension=IntentDimension(   # 新增
                        id=BuiltinDimensions.DIALOGUE,
                        direction=IntentDirection.INCREASE,
                    ),
                    desired_effect="Test intent",
                    priority=IntentPriority.MEDIUM,
                )
            ]
        return []


class TestIntentCompiler:
    def test_compiles_intents_from_analyzers(self):
        compiler = IntentCompiler(analyzers=[MockAnalyzer()])
        obs = MockObservation({})
        result = compiler.compile(obs, {})

        assert len(result.intents) == 1
        assert result.intents[0].desired_effect == "Test intent"

    def test_returns_empty_when_no_analyzers(self):
        compiler = IntentCompiler(analyzers=[])
        obs = MockObservation({})
        result = compiler.compile(obs, {})

        assert len(result.intents) == 0

    def test_handles_analyzer_failure_gracefully(self):
        class FailingAnalyzer(IntentAnalyzer):
            def analyze(self, observation, context):
                raise ValueError("Simulated failure")

        compiler = IntentCompiler(analyzers=[FailingAnalyzer(), MockAnalyzer()])
        obs = MockObservation({})
        result = compiler.compile(obs, {})

        # 即使一个 Analyzer 失败，其他 Analyzer 仍可工作
        assert len(result.intents) == 1