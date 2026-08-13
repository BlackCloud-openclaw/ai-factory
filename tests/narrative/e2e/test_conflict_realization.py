# tests/narrative/e2e/test_conflict_realization.py

import pytest
from unittest.mock import AsyncMock

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
    IntentDirection,
    BuiltinDimensions,
    IntentResolver,
)
from src.narrative.resolution.context_builder import enrich_narrative_context
from src.narrative.realizers.reference import ReferenceNarrativeRealizer
from src.narrative.realizers.interfaces import TextGenerator
from src.narrative.loops.quality_loop import QualityLoop
from src.narrative.intent import IntentSatisfaction, EvaluationResult
from src.narrative.conflict import create_resolver


class ConflictMockTextGenerator(TextGenerator):
    def __init__(self, expected_strategy: str):
        self.last_prompt = ""
        self.expected = expected_strategy

    async def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        assert f"冲突策略: {self.expected}" in prompt
        return f"{self.expected} 策略生成"


class NoConflictMockTextGenerator(TextGenerator):
    async def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        assert "冲突策略" not in prompt
        return "无冲突生成"


class AlwaysPassEvaluator:
    async def evaluate(self, artifact, intent) -> EvaluationResult:
        return EvaluationResult(score=1.0, reason="Mock always pass", evaluator="Mock")


@pytest.mark.asyncio
async def test_conflict_resolution_pipeline():
    inc = NarrativeIntent(
        source=IntentSource.SYSTEM,
        dimension=IntentDimension(
            id=BuiltinDimensions.DIALOGUE,
            direction=IntentDirection.INCREASE,
        ),
        desired_effect="增加对白，提升角色互动",
        priority=IntentPriority.HIGH,
    )
    dec = NarrativeIntent(
        source=IntentSource.SYSTEM,
        dimension=IntentDimension(
            id=BuiltinDimensions.DIALOGUE,
            direction=IntentDirection.DECREASE,  # 相反方向
        ),
        desired_effect="减少冗余对白，保持节奏",
        priority=IntentPriority.LOW,
    )
    intents = NarrativeIntentSet(intents=(inc, dec))

    resolver = IntentResolver()
    plan = resolver.resolve(intents)

    assert len(plan.conflicts) == 1
    assert len(plan.resolutions) == 1
    res = plan.resolutions[0]
    assert res.strategy.value == "priority"
    assert res.selected_intent == inc.id

    meta = ChapterMetadata(volume=1, chapter=1, scene_index=0, total_scenes=3)
    story = StorySnapshot(projection={"chapter": 1})
    context = NarrativeContext(story=story, metadata=meta)
    context = enrich_narrative_context(context, plan)
    assert context.resolution_context is not None
    assert len(context.resolution_context.resolutions) == 1

    generator = ConflictMockTextGenerator("priority")
    realizer = ReferenceNarrativeRealizer(generator)

    mock_satisfaction = IntentSatisfaction(
        evaluator=AlwaysPassEvaluator(),
        threshold=0.0,
    )
    loop = QualityLoop(
        realizer=realizer,
        satisfaction=mock_satisfaction,
        max_iterations=1,
        acceptance_threshold=0.0,
    )

    artifact = NarrativeArtifact(text="原始草稿文本")
    constraint = NarrativeConstraint()

    result = await loop.run(artifact, context, plan, constraint)

    assert result.accepted is True
    assert result.iterations == 1
    assert context.resolution_context.resolutions == result.plan.resolutions
    assert "priority" in generator.last_prompt
    assert result.artifact.text == "priority 策略生成"


@pytest.mark.asyncio
async def test_balance_e2e():
    inc = NarrativeIntent(
        source=IntentSource.SYSTEM,
        dimension=IntentDimension(
            id=BuiltinDimensions.DIALOGUE,
            direction=IntentDirection.INCREASE,
        ),
        desired_effect="增加对话深度",
        priority=IntentPriority.HIGH,
    )
    dec = NarrativeIntent(
        source=IntentSource.SYSTEM,
        dimension=IntentDimension(
            id=BuiltinDimensions.DIALOGUE,
            direction=IntentDirection.DECREASE,  # 相反方向
        ),
        desired_effect="减少无效对白",
        priority=IntentPriority.HIGH,
    )
    intents = NarrativeIntentSet(intents=(inc, dec))

    resolver = IntentResolver(conflict_resolver=create_resolver("balance"))
    plan = resolver.resolve(intents)

    assert len(plan.resolutions) == 1
    assert plan.resolutions[0].strategy.value == "balance"
    assert plan.resolutions[0].selected_intent is None

    meta = ChapterMetadata(volume=1, chapter=1, scene_index=0, total_scenes=3)
    story = StorySnapshot(projection={})
    context = NarrativeContext(story=story, metadata=meta)
    context = enrich_narrative_context(context, plan)

    generator = ConflictMockTextGenerator("balance")
    realizer = ReferenceNarrativeRealizer(generator)

    mock_satisfaction = IntentSatisfaction(
        evaluator=AlwaysPassEvaluator(),
        threshold=0.0,
    )
    loop = QualityLoop(
        realizer=realizer,
        satisfaction=mock_satisfaction,
        max_iterations=1,
        acceptance_threshold=0.0,
    )

    artifact = NarrativeArtifact(text="原始草稿")
    constraint = NarrativeConstraint()
    result = await loop.run(artifact, context, plan, constraint)

    assert result.accepted is True
    assert result.artifact.text == "balance 策略生成"
    assert "冲突策略: balance" in generator.last_prompt
    assert "平衡双方目标" in generator.last_prompt


@pytest.mark.asyncio
async def test_synthesis_e2e():
    inc = NarrativeIntent(
        source=IntentSource.SYSTEM,
        dimension=IntentDimension(
            id=BuiltinDimensions.DIALOGUE,
            direction=IntentDirection.INCREASE,
        ),
        desired_effect="增加互动",
        priority=IntentPriority.MEDIUM,
    )
    dec = NarrativeIntent(
        source=IntentSource.SYSTEM,
        dimension=IntentDimension(
            id=BuiltinDimensions.DIALOGUE,
            direction=IntentDirection.DECREASE,  # 相反方向
        ),
        desired_effect="精简节奏",
        priority=IntentPriority.MEDIUM,
    )
    intents = NarrativeIntentSet(intents=(inc, dec))

    resolver = IntentResolver(conflict_resolver=create_resolver("synthesis"))
    plan = resolver.resolve(intents)

    assert len(plan.resolutions) == 1
    assert plan.resolutions[0].strategy.value == "synthesis"
    assert plan.resolutions[0].selected_intent is None

    meta = ChapterMetadata(volume=1, chapter=1, scene_index=0, total_scenes=3)
    story = StorySnapshot(projection={})
    context = NarrativeContext(story=story, metadata=meta)
    context = enrich_narrative_context(context, plan)

    generator = ConflictMockTextGenerator("synthesis")
    realizer = ReferenceNarrativeRealizer(generator)

    mock_satisfaction = IntentSatisfaction(
        evaluator=AlwaysPassEvaluator(),
        threshold=0.0,
    )
    loop = QualityLoop(
        realizer=realizer,
        satisfaction=mock_satisfaction,
        max_iterations=1,
        acceptance_threshold=0.0,
    )

    artifact = NarrativeArtifact(text="原始草稿")
    constraint = NarrativeConstraint()
    result = await loop.run(artifact, context, plan, constraint)

    assert result.accepted is True
    assert result.artifact.text == "synthesis 策略生成"
    assert "冲突策略: synthesis" in generator.last_prompt
    assert "更高层面" in generator.last_prompt or "统一" in generator.last_prompt


@pytest.mark.asyncio
async def test_no_conflict_pipeline():
    i1 = NarrativeIntent(
        source=IntentSource.SYSTEM,
        dimension=IntentDimension.dialogue(),
        desired_effect="增加对白",
    )
    i2 = NarrativeIntent(
        source=IntentSource.SYSTEM,
        dimension=IntentDimension.emotion(),
        desired_effect="增强情绪",
    )
    intents = NarrativeIntentSet(intents=(i1, i2))

    resolver = IntentResolver()
    plan = resolver.resolve(intents)

    assert len(plan.conflicts) == 0
    assert len(plan.resolutions) == 0

    meta = ChapterMetadata(volume=1, chapter=1, scene_index=0, total_scenes=3)
    story = StorySnapshot(projection={})
    context = NarrativeContext(story=story, metadata=meta)
    context = enrich_narrative_context(context, plan)
    assert context.resolution_context is None

    generator = NoConflictMockTextGenerator()
    realizer = ReferenceNarrativeRealizer(generator)

    mock_satisfaction = IntentSatisfaction(
        evaluator=AlwaysPassEvaluator(),
        threshold=0.0,
    )
    loop = QualityLoop(
        realizer=realizer,
        satisfaction=mock_satisfaction,
        max_iterations=1,
        acceptance_threshold=0.0,
    )

    artifact = NarrativeArtifact(text="原始草稿")
    constraint = NarrativeConstraint()
    result = await loop.run(artifact, context, plan, constraint)

    assert result.accepted is True
    assert context.resolution_context is None
    assert result.artifact.text == "无冲突生成"
    assert "冲突策略" not in generator.last_prompt