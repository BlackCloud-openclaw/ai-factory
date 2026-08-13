# tests/narrative/test_realizer.py

import pytest
import inspect

from src.narrative import (
    NarrativeRealizer,
    NarrativeArtifact,
    NarrativeContext,
    NarrativeIntent,
    NarrativeIntentSet,
    NarrativeConstraint,
    IntentSource,
    IntentPriority,
    IntentDimension,
    IntentDirection,
    BuiltinDimensions,
    StorySnapshot,
    ChapterMetadata,
    ResolutionPlan,
)


class TestNarrativeRealizerProtocol:
    def test_protocol_runtime_checkable(self):
        class MockRealizer:
            async def realize(
                self,
                artifact: NarrativeArtifact,
                context: NarrativeContext,
                plan: ResolutionPlan,  # ✅ 修正参数名
                constraint: NarrativeConstraint,
            ) -> NarrativeArtifact:
                return artifact

        mock = MockRealizer()
        assert isinstance(mock, NarrativeRealizer)

    def test_protocol_accepts_any_implementation(self):
        class RealizerImpl:
            async def realize(
                self,
                artifact: NarrativeArtifact,
                context: NarrativeContext,
                plan: ResolutionPlan,  # ✅ 修正参数名
                constraint: NarrativeConstraint,
            ) -> NarrativeArtifact:
                return NarrativeArtifact(text=artifact.text + " edited")

        impl = RealizerImpl()

        artifact = NarrativeArtifact(text="original")
        story = StorySnapshot(projection={})
        meta = ChapterMetadata(volume=1, chapter=1, scene_index=0, total_scenes=3)
        context = NarrativeContext(story=story, metadata=meta)

        # ✅ 创建包含 dimension 的 Intent
        intent = NarrativeIntent(
            source=IntentSource.EDITORIAL,
            dimension=IntentDimension(
                id=BuiltinDimensions.DIALOGUE,
                direction=IntentDirection.INCREASE,
            ),
            desired_effect="improve readability",
            priority=IntentPriority.HIGH,
        )
        intents = NarrativeIntentSet(intents=(intent,))
        plan = ResolutionPlan(primary_intents=(intent,))
        constraint = NarrativeConstraint()

        import asyncio
        result = asyncio.run(impl.realize(artifact, context, plan, constraint))
        assert result.text == "original edited"

    def test_protocol_signature(self):
        sig = inspect.signature(NarrativeRealizer.realize)
        params = list(sig.parameters.values())

        assert len(params) == 5
        assert params[0].name == "self"
        assert params[1].name == "artifact"
        assert params[2].name == "context"
        assert params[3].name == "plan"      # ✅ 修正：plan 而不是 intents
        assert params[4].name == "constraint"

        assert params[1].annotation == NarrativeArtifact
        assert params[2].annotation == NarrativeContext
        assert params[3].annotation == ResolutionPlan
        assert params[4].annotation == NarrativeConstraint

        assert sig.return_annotation == NarrativeArtifact