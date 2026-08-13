# tests/narrative/loops/test_quality_loop.py

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.narrative import (
    NarrativeArtifact,
    NarrativeContext,
    ChapterMetadata,
    StorySnapshot,
    NarrativeConstraint,
    NarrativeIntent,
    NarrativeIntentSet,
    IntentSource,
    IntentPriority,
    IntentDimension,
    BuiltinDimensions,
    IntentDirection,
    ResolutionPlan,
    IntentResolver,
)
from src.narrative.loops.quality_loop import QualityLoop
from src.narrative.intent import IntentSatisfaction, EvaluationResult


# 模拟满意度评估器：始终返回 1.0 且 passed=True
class AlwaysPassEvaluator:
    async def evaluate(self, artifact, intent) -> EvaluationResult:
        return EvaluationResult(score=1.0, reason="Mock always pass", evaluator="Mock")


class MockRealizer:
    def __init__(self, response: str = "edited text"):
        self.response = response
        self.call_count = 0

    async def realize(self, artifact, context, plan, constraint):
        self.call_count += 1
        return NarrativeArtifact(text=self.response)


class TestQualityLoop:
    @pytest.mark.asyncio
    async def test_loop_accepts_on_first_iteration(self):
        realizer = MockRealizer("good text")
        # 使用模拟评估器，确保第一次迭代即通过
        mock_satisfaction = IntentSatisfaction(
            evaluator=AlwaysPassEvaluator(),
            threshold=0.0,
        )
        loop = QualityLoop(
            realizer=realizer,
            satisfaction=mock_satisfaction,
            acceptance_threshold=0.0,
        )

        intent = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension.dialogue(),
            desired_effect="增强人物互动",
        )
        intents = NarrativeIntentSet(intents=(intent,))
        resolver = IntentResolver()
        plan = resolver.resolve(intents)

        artifact = NarrativeArtifact(text="original text")
        meta = ChapterMetadata(volume=1, chapter=1, scene_index=0, total_scenes=3)
        context = NarrativeContext(story=StorySnapshot(projection={}), metadata=meta)
        constraint = NarrativeConstraint()

        result = await loop.run(artifact, context, plan, constraint)

        assert result.accepted is True
        assert result.iterations == 1
        assert result.artifact.text == "good text"
        assert result.plan.primary_intents is not None

    @pytest.mark.asyncio
    async def test_loop_iterates_max_iterations(self):
        realizer = MockRealizer("still not good")
        # 使用真实评估器（关键词匹配），因测试文本无匹配，所以会迭代满次数
        loop = QualityLoop(
            realizer=realizer,
            acceptance_threshold=0.99,
            max_iterations=3,
        )

        intent = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension.dialogue(),
            desired_effect="非常高要求的编辑",
        )
        intents = NarrativeIntentSet(intents=(intent,))
        resolver = IntentResolver()
        plan = resolver.resolve(intents)

        artifact = NarrativeArtifact(text="original")
        meta = ChapterMetadata(volume=1, chapter=1, scene_index=0, total_scenes=3)
        context = NarrativeContext(story=StorySnapshot(projection={}), metadata=meta)
        constraint = NarrativeConstraint()

        result = await loop.run(artifact, context, plan, constraint)

        assert result.accepted is False
        assert result.max_iterations_reached is True
        assert result.iterations == 3