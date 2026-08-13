# tests/narrative/realizers/test_reference.py

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
)
from src.narrative.realizers.reference import ReferenceNarrativeRealizer
from src.narrative.realizers.interfaces import TextGenerator


class MockTextGenerator:
    def __init__(self, response: str = "edited text"):
        self.response = response

    async def generate(self, prompt: str) -> str:
        return self.response


class TestReferenceNarrativeRealizer:
    def test_instantiation(self):
        generator = MockTextGenerator()
        realizer = ReferenceNarrativeRealizer(generator)
        assert realizer._text_generator is generator

    def test_implements_protocol(self):
        from src.narrative.realizer import NarrativeRealizer
        generator = MockTextGenerator()
        realizer = ReferenceNarrativeRealizer(generator)
        assert isinstance(realizer, NarrativeRealizer)

    @pytest.mark.asyncio
    async def test_no_intent_returns_original(self):
        generator = MockTextGenerator()
        realizer = ReferenceNarrativeRealizer(generator)

        artifact = NarrativeArtifact(text="original text")
        story = StorySnapshot(projection={})
        meta = ChapterMetadata(volume=1, chapter=1, scene_index=0, total_scenes=3)
        context = NarrativeContext(story=story, metadata=meta)
        # ✅ 传入 ResolutionPlan 而不是 NarrativeIntentSet
        plan = ResolutionPlan(primary_intents=())
        constraint = NarrativeConstraint()

        result = await realizer.realize(artifact, context, plan, constraint)
        assert result.text == "original text"

    @pytest.mark.asyncio
    async def test_realize_calls_generator(self):
        generator = MockTextGenerator("edited text")
        realizer = ReferenceNarrativeRealizer(generator)

        artifact = NarrativeArtifact(text="original text")
        story = StorySnapshot(projection={})
        meta = ChapterMetadata(volume=1, chapter=1, scene_index=0, total_scenes=3)
        context = NarrativeContext(story=story, metadata=meta)

        # ✅ 创建包含 dimension 的 Intent
        intent = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension(
                id=BuiltinDimensions.DIALOGUE,
                direction=IntentDirection.INCREASE,
            ),
            desired_effect="improve readability",
            priority=IntentPriority.HIGH,
        )
        plan = ResolutionPlan(primary_intents=(intent,))
        constraint = NarrativeConstraint()

        result = await realizer.realize(artifact, context, plan, constraint)
        assert result.text == "edited text"

    @pytest.mark.asyncio
    async def test_realize_fallback_on_error(self):
        class FailingGenerator:
            async def generate(self, prompt: str) -> str:
                raise Exception("Simulated failure")

        generator = FailingGenerator()
        realizer = ReferenceNarrativeRealizer(generator)

        artifact = NarrativeArtifact(text="original")
        story = StorySnapshot(projection={})
        meta = ChapterMetadata(volume=1, chapter=1, scene_index=0, total_scenes=3)
        context = NarrativeContext(story=story, metadata=meta)
        intent = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension.dialogue(),
            desired_effect="improve",
        )
        plan = ResolutionPlan(primary_intents=(intent,))
        constraint = NarrativeConstraint()

        result = await realizer.realize(artifact, context, plan, constraint)
        assert result.text == "original"

    @pytest.mark.asyncio
    async def test_prompt_contains_intent_desired_effect(self):
        generator = MockTextGenerator("edited text")
        realizer = ReferenceNarrativeRealizer(generator)

        artifact = NarrativeArtifact(text="original")
        story = StorySnapshot(projection={})
        meta = ChapterMetadata(volume=1, chapter=1, scene_index=0, total_scenes=3)
        context = NarrativeContext(story=story, metadata=meta)

        intent = NarrativeIntent(
            source=IntentSource.EDITORIAL,
            dimension=IntentDimension.dialogue(),
            desired_effect="Make the reader feel tension during the duel",
            preserve=("good pacing",),
            avoid=("info-dump",),
            priority=IntentPriority.HIGH,
        )
        plan = ResolutionPlan(primary_intents=(intent,))
        constraint = NarrativeConstraint()

        # 捕获内部生成的 prompt
        realizer._text_generator = MockTextGenerator("edited")
        await realizer.realize(artifact, context, plan, constraint)

        # 无法直接获取 prompt，但可以通过测试 build_editor_prompt 间接验证
        from src.narrative.realizers.prompts import build_editor_prompt
        prompt = build_editor_prompt(
            artifact_text=artifact.text,
            context=context,
            intents=[intent],
            constraint_summary="- 保持剧情事实不变",
        )
        assert "Make the reader feel tension during the duel" in prompt

    @pytest.mark.asyncio
    async def test_prompt_priority_order(self):
        intent_low = NarrativeIntent(
            source=IntentSource.EDITORIAL,
            dimension=IntentDimension.dialogue(),
            desired_effect="Low priority item",
            priority=IntentPriority.LOW,
        )
        intent_high = NarrativeIntent(
            source=IntentSource.EDITORIAL,
            dimension=IntentDimension.dialogue(),
            desired_effect="High priority item",
            priority=IntentPriority.HIGH,
        )

        # 验证 Prompt 中 High 出现在 Low 之前
        from src.narrative.realizers.prompts import build_editor_prompt
        context = NarrativeContext(
            story=StorySnapshot(projection={}),
            metadata=ChapterMetadata(volume=1, chapter=1, scene_index=0, total_scenes=3),
        )
        prompt = build_editor_prompt(
            artifact_text="test",
            context=context,
            intents=[intent_low, intent_high],
            constraint_summary="- 保持剧情事实不变",
        )

        high_pos = prompt.find("High priority item")
        low_pos = prompt.find("Low priority item")
        assert high_pos < low_pos

    def test_text_generator_protocol(self):
        class CustomGenerator:
            async def generate(self, prompt: str) -> str:
                return "test"

        generator = CustomGenerator()
        assert isinstance(generator, TextGenerator)