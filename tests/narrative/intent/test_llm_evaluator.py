# tests/narrative/intent/test_llm_evaluator.py

import pytest
from unittest.mock import AsyncMock

from src.narrative.artifact import NarrativeArtifact
from src.narrative.intent import (
    NarrativeIntent,
    IntentSource,
    IntentDimension,
    LLMSemanticEvaluator,
    EvaluationResult,
)
from src.narrative.realizers.interfaces import TextGenerator


class MockTextGenerator:
    def __init__(self, response: str):
        self.response = response
        self.call_count = 0
        self.last_prompt = ""

    async def generate(self, prompt: str) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        return self.response


class TestLLMSemanticEvaluator:
    @pytest.mark.asyncio
    async def test_semantic_eval_success(self):
        mock = MockTextGenerator(
            '{"score": 85, "reason": "冲突明显升级", "evidence": ["对峙场景"]}'
        )
        evaluator = LLMSemanticEvaluator(mock)

        artifact = NarrativeArtifact(text="两人在雨中对峙，剑拔弩张。")
        intent = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension.dialogue(),
            desired_effect="增加冲突张力",
        )

        result = await evaluator.evaluate(artifact, intent)

        assert isinstance(result, EvaluationResult)
        assert result.score == 0.85
        assert result.reason == "冲突明显升级"
        assert result.evidence == ("对峙场景",)
        assert result.evaluator == "LLM"
        assert result.fallback is False
        assert mock.call_count == 1
        assert "增加冲突张力" in mock.last_prompt

    @pytest.mark.asyncio
    async def test_semantic_eval_json_error_fallback(self):
        mock = MockTextGenerator("这不是 JSON")
        fallback = AsyncMock()
        fallback.evaluate.return_value = EvaluationResult(
            score=0.7,
            reason="Fallback triggered",
            evaluator="Keyword",
            fallback=True,
        )

        evaluator = LLMSemanticEvaluator(mock, fallback_evaluator=fallback)

        artifact = NarrativeArtifact(text="test")
        intent = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension.dialogue(),
            desired_effect="test",
        )

        result = await evaluator.evaluate(artifact, intent)

        assert result.score == 0.7
        assert result.fallback is True
        assert result.evaluator == "Keyword"
        fallback.evaluate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_semantic_eval_markdown_cleaning(self):
        mock = MockTextGenerator(
            '```json\n{"score": 90, "reason": "很好", "evidence": ["a", "b"]}\n```'
        )
        evaluator = LLMSemanticEvaluator(mock)

        artifact = NarrativeArtifact(text="test")
        intent = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension.dialogue(),
            desired_effect="test",
        )

        result = await evaluator.evaluate(artifact, intent)

        assert result.score == 0.9

    @pytest.mark.asyncio
    async def test_empty_intent(self):
        mock = MockTextGenerator("")  # 不会被调用
        evaluator = LLMSemanticEvaluator(mock)

        artifact = NarrativeArtifact(text="test")
        intent = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension.dialogue(),
            desired_effect="",
        )

        result = await evaluator.evaluate(artifact, intent)

        assert result.score == 1.0
        assert mock.call_count == 0  # LLM 未调用

    @pytest.mark.asyncio
    async def test_score_boundary(self):
        mock = MockTextGenerator('{"score": 150, "reason": "too high"}')
        evaluator = LLMSemanticEvaluator(mock)

        artifact = NarrativeArtifact(text="test")
        intent = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension.dialogue(),
            desired_effect="test",
        )

        result = await evaluator.evaluate(artifact, intent)
        assert result.score == 1.0

        mock.response = '{"score": -10}'
        result = await evaluator.evaluate(artifact, intent)
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_prompt_contains_dimension(self):
        mock = MockTextGenerator('{"score": 50}')
        evaluator = LLMSemanticEvaluator(mock)

        artifact = NarrativeArtifact(text="test")
        intent = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension.dialogue(),
            desired_effect="增加对话",
        )

        await evaluator.evaluate(artifact, intent)
        assert "维度: narrative.dialogue" in mock.last_prompt
        assert "方向: increase" in mock.last_prompt