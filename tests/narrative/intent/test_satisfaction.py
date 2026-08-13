# tests/narrative/intent/test_satisfaction.py

import pytest
from unittest.mock import AsyncMock

from src.narrative.artifact import NarrativeArtifact
from src.narrative.intent import (
    NarrativeIntent,
    NarrativeIntentSet,
    IntentSource,
    IntentPriority,
    IntentDimension,
    IntentDirection,
    BuiltinDimensions,
    KeywordSatisfactionEvaluator,
    IntentSatisfaction,
    SatisfactionEvaluator,
    SatisfactionReport,
    evaluate_satisfaction,
    EvaluationResult,          # 新增
)


class TestKeywordSatisfactionEvaluator:
    @pytest.mark.asyncio
    async def test_matching_text(self):
        evaluator = KeywordSatisfactionEvaluator()
        artifact = NarrativeArtifact(
            text="林逸和长老进行了长时间的对话，两人在灵泉边互动交流。"
        )
        intent = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension.dialogue(),
            desired_effect="互动",
        )

        result = await evaluator.evaluate(artifact, intent)
        assert isinstance(result, EvaluationResult)
        assert result.score > 0.0
        assert result.evaluator == "Keyword"
        assert result.fallback is False
        assert "互动" in result.evidence

    @pytest.mark.asyncio
    async def test_non_matching_text(self):
        evaluator = KeywordSatisfactionEvaluator()
        artifact = NarrativeArtifact(
            text="林逸独自在丹房修炼了三天三夜。"
        )
        intent = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension.dialogue(),
            desired_effect="互动",
        )

        result = await evaluator.evaluate(artifact, intent)
        assert result.score == 0.0
        assert result.evidence == ()

    @pytest.mark.asyncio
    async def test_empty_desired_effect(self):
        evaluator = KeywordSatisfactionEvaluator()
        artifact = NarrativeArtifact(text="test")
        intent = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension.dialogue(),
            desired_effect="",
        )

        result = await evaluator.evaluate(artifact, intent)
        assert result.score == 1.0
        assert result.reason == "意图为空，视为已满足"


class TestIntentSatisfaction:
    @pytest.mark.asyncio
    async def test_evaluate_with_keyword_evaluator(self):
        satisfaction = IntentSatisfaction(threshold=0.3)
        artifact = NarrativeArtifact(
            text="林逸和长老在灵泉边对话，互动生动。"
        )
        intent = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension.dialogue(),
            desired_effect="互动",
        )
        intents = NarrativeIntentSet(intents=(intent,))

        report = await satisfaction.evaluate(artifact, intents)

        assert report.overall > 0.0
        assert len(report.items) == 1
        assert report.items[0].intent_id == str(intent.id)
        assert report.items[0].score > 0.0
        assert "互动" in report.items[0].evidence

    @pytest.mark.asyncio
    async def test_empty_intents(self):
        satisfaction = IntentSatisfaction()
        artifact = NarrativeArtifact(text="test")
        intents = NarrativeIntentSet()

        report = await satisfaction.evaluate(artifact, intents)

        assert report.overall == 1.0
        assert report.passed is True
        assert len(report.items) == 0
        assert report.metadata.get("reason") == "no_intents"

    @pytest.mark.asyncio
    async def test_multiple_intents(self):
        satisfaction = IntentSatisfaction()
        artifact = NarrativeArtifact(
            text="林逸和长老在灵泉边对话，情绪激动。场景自然过渡到下一个画面。"
        )
        i1 = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension.dialogue(),
            desired_effect="对话",
        )
        i2 = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension.emotion(),
            desired_effect="情绪",
        )
        intents = NarrativeIntentSet(intents=(i1, i2))

        report = await satisfaction.evaluate(artifact, intents)

        assert len(report.items) == 2
        assert report.items[0].score is not None
        assert report.items[1].score is not None
        # 不强制通过或失败，只检查结构
        assert report.passed in (True, False)

    @pytest.mark.asyncio
    async def test_report_serialization(self):
        satisfaction = IntentSatisfaction()
        artifact = NarrativeArtifact(text="test dialogue")
        intent = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension.dialogue(),
            desired_effect="对话",
        )
        intents = NarrativeIntentSet(intents=(intent,))

        report = await satisfaction.evaluate(artifact, intents)
        data = report.to_dict()

        assert "overall" in data
        assert "passed" in data
        assert "items" in data
        assert "metadata" in data
        assert len(data["items"]) == 1
        assert data["items"][0]["intent_id"] == str(intent.id)


class TestCustomEvaluator:
    """验证可插拔评估器机制（必须返回 EvaluationResult）"""

    class CustomEvaluator:
        async def evaluate(self, artifact, intent) -> EvaluationResult:
            return EvaluationResult(
                score=0.85,
                reason="自定义评估",
                evidence=("custom",),
                evaluator="Custom",
                fallback=False,
            )

    @pytest.mark.asyncio
    async def test_custom_evaluator(self):
        custom = self.CustomEvaluator()
        satisfaction = IntentSatisfaction(evaluator=custom, threshold=0.5)
        artifact = NarrativeArtifact(text="test")
        intent = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension.dialogue(),
            desired_effect="test",
        )
        intents = NarrativeIntentSet(intents=(intent,))

        report = await satisfaction.evaluate(artifact, intents)

        assert report.overall == 0.85
        assert report.items[0].score == 0.85
        assert report.items[0].evidence == ("custom",)
        assert report.items[0].reason == "自定义评估"
        assert report.passed is True
        assert "CustomEvaluator" in report.metadata.get("evaluator_type", "")

    def test_report_metadata_immutable(self):
        """验证 SatisfactionReport.metadata 不可变"""
        report = SatisfactionReport(overall=1.0, metadata={"a": 1})
        with pytest.raises(TypeError):
            report.metadata["b"] = 2  # type: ignore


class TestEvaluateSatisfactionConvenience:
    @pytest.mark.asyncio
    async def test_convenience_function(self):
        artifact = NarrativeArtifact(text="test dialogue")
        intent = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension.dialogue(),
            desired_effect="对话",
        )
        intents = NarrativeIntentSet(intents=(intent,))

        report = await evaluate_satisfaction(artifact, intents)

        assert report.overall is not None
        assert len(report.items) == 1
        assert report.items[0].score is not None