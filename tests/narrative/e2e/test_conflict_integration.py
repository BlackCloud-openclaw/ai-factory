# tests/narrative/e2e/test_conflict_integration.py

import pytest

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
    ResolutionPlan,
)
from src.narrative.intent import IntentResolver
from src.narrative.realizers.reference import ReferenceNarrativeRealizer


class RecordingTextGenerator:
    def __init__(self, response: str = "edited text"):
        self.response = response
        self.prompt = ""

    async def generate(self, prompt: str) -> str:
        self.prompt = prompt
        return self.response


class TestConflictIntegration:
    @pytest.mark.asyncio
    async def test_conflict_detected_in_resolution_plan(self):
        inc = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension(
                id=BuiltinDimensions.DIALOGUE,
                direction=IntentDirection.INCREASE,
            ),
            desired_effect="增加对白数量",
            priority=IntentPriority.HIGH,
        )
        dec = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension(
                id=BuiltinDimensions.DIALOGUE,
                direction=IntentDirection.DECREASE,
            ),
            desired_effect="减少对白数量",
            priority=IntentPriority.HIGH,
        )
        intents = NarrativeIntentSet(intents=(inc, dec))

        resolver = IntentResolver()
        plan = resolver.resolve(intents)

        assert plan.has_conflicts is True
        assert plan.conflict_count == 1
        assert len(plan.primary_intents) == 2

    @pytest.mark.asyncio
    async def test_resolution_plan_passed_to_realizer(self):
        inc = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension(
                id=BuiltinDimensions.DIALOGUE,
                direction=IntentDirection.INCREASE,
            ),
            desired_effect="增加对白数量",
        )
        dec = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension(
                id=BuiltinDimensions.DIALOGUE,
                direction=IntentDirection.DECREASE,
            ),
            desired_effect="减少对白数量",
        )
        intents = NarrativeIntentSet(intents=(inc, dec))

        resolver = IntentResolver()
        plan = resolver.resolve(intents)

        assert plan.has_conflicts is True

        generator = RecordingTextGenerator("edited text")
        realizer = ReferenceNarrativeRealizer(generator)

        artifact = NarrativeArtifact(text="original")
        meta = ChapterMetadata(volume=1, chapter=1, scene_index=0, total_scenes=3)
        context = NarrativeContext(
            story=StorySnapshot(projection={"chapter": 1}),
            metadata=meta,
        )
        constraint = NarrativeConstraint()

        result = await realizer.realize(artifact, context, plan, constraint)

        assert result.text == "edited text"
        assert "增加对白" in generator.prompt or "减少对白" in generator.prompt

    @pytest.mark.asyncio
    async def test_no_conflict_plan(self):
        i1 = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension(
                id=BuiltinDimensions.DIALOGUE,
                direction=IntentDirection.INCREASE,
            ),
            desired_effect="增加对白",
        )
        i2 = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension(
                id=BuiltinDimensions.EMOTION,
                direction=IntentDirection.INCREASE,
            ),
            desired_effect="增强情绪",
        )
        intents = NarrativeIntentSet(intents=(i1, i2))

        resolver = IntentResolver()
        plan = resolver.resolve(intents)

        assert plan.has_conflicts is False
        assert plan.conflict_count == 0

        generator = RecordingTextGenerator("no conflict text")
        realizer = ReferenceNarrativeRealizer(generator)

        artifact = NarrativeArtifact(text="original")
        meta = ChapterMetadata(volume=1, chapter=1, scene_index=0, total_scenes=3)
        context = NarrativeContext(
            story=StorySnapshot(projection={"chapter": 1}),
            metadata=meta,
        )
        constraint = NarrativeConstraint()

        result = await realizer.realize(artifact, context, plan, constraint)
        assert result.text == "no conflict text"