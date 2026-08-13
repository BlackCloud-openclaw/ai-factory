# tests/narrative/e2e/test_realizer_pipeline.py

import pytest

from src.narrative.artifact import NarrativeArtifact
from src.narrative.context import NarrativeContext, ChapterMetadata
from src.narrative.constraint import NarrativeConstraint
from src.narrative.snapshot import StorySnapshot
from src.narrative.intent import IntentPriority, IntentResolver
from src.narrative.compilers.intent_compiler import IntentCompiler
from src.narrative.compilers.analyzers import DialogueAnalyzer, TransitionAnalyzer, EmotionAnalyzer
from src.narrative.adapters.observation_adapter import ObservationAdapter
from src.narrative.realizers.reference import ReferenceNarrativeRealizer


class RecordingTextGenerator:
    def __init__(self, response: str = "edited chapter content"):
        self.response = response
        self.prompt = ""

    async def generate(self, prompt: str) -> str:
        self.prompt = prompt
        return self.response


class TestRealizerPipeline:
    @pytest.mark.asyncio
    async def test_observation_to_artifact(self):
        observation_data = {
            "dimensions": {
                "dialogue_ratio": 0.10,
                "transition_score": 0.30,
                "emotion_score": 0.20,
            }
        }
        observation = ObservationAdapter(observation_data)

        compiler = IntentCompiler(
            analyzers=[
                DialogueAnalyzer(min_ratio=0.20),
                TransitionAnalyzer(min_score=0.5),
                EmotionAnalyzer(min_score=0.4),
            ]
        )

        context = {"volume": 1, "chapter": 1, "scene_index": 0, "total_scenes": 3}
        intents = compiler.compile(observation, context)
        assert len(intents.intents) == 3

        # ✅ 创建 Resolver 并解析为 ResolutionPlan
        resolver = IntentResolver()
        plan = resolver.resolve(intents)

        generator = RecordingTextGenerator("edited chapter content")
        realizer = ReferenceNarrativeRealizer(generator)

        artifact = NarrativeArtifact(text="original draft text")
        meta = ChapterMetadata(volume=1, chapter=1, scene_index=0, total_scenes=3)
        narrative_context = NarrativeContext(
            story=StorySnapshot(projection={"chapter": 1, "events": ["event_1"]}),
            metadata=meta,
        )
        constraint = NarrativeConstraint()

        # ✅ 传入 plan 而不是 intents
        result = await realizer.realize(artifact, narrative_context, plan, constraint)

        assert result.text == "edited chapter content"
        assert result.artifact_id == artifact.artifact_id

    @pytest.mark.asyncio
    async def test_intent_in_prompt(self):
        observation_data = {
            "dimensions": {
                "dialogue_ratio": 0.10,
                "transition_score": 0.30,
                "emotion_score": 0.20,
            }
        }
        observation = ObservationAdapter(observation_data)

        compiler = IntentCompiler(
            analyzers=[
                DialogueAnalyzer(min_ratio=0.20),
                TransitionAnalyzer(min_score=0.5),
                EmotionAnalyzer(min_score=0.4),
            ]
        )

        context = {"volume": 1, "chapter": 1, "scene_index": 0, "total_scenes": 3}
        intents = compiler.compile(observation, context)

        resolver = IntentResolver()
        plan = resolver.resolve(intents)

        generator = RecordingTextGenerator("edited chapter content")
        realizer = ReferenceNarrativeRealizer(generator)

        artifact = NarrativeArtifact(text="original draft text")
        meta = ChapterMetadata(volume=1, chapter=1, scene_index=0, total_scenes=3)
        narrative_context = NarrativeContext(
            story=StorySnapshot(projection={"chapter": 1}),
            metadata=meta,
        )
        constraint = NarrativeConstraint()

        # ✅ 传入 plan
        await realizer.realize(artifact, narrative_context, plan, constraint)

        assert "增强人物互动" in generator.prompt
        assert "场景之间的过渡" in generator.prompt
        assert "情绪表达" in generator.prompt

    @pytest.mark.asyncio
    async def test_analyzer_priority_order(self):
        from src.narrative.compilers.analyzers.transition import TransitionAnalyzer
        from src.narrative.compilers.analyzers.dialogue import DialogueAnalyzer

        compiler = IntentCompiler(
            analyzers=[
                TransitionAnalyzer(min_score=0.8),
                DialogueAnalyzer(min_ratio=0.30),
            ]
        )

        observation_data = {
            "dimensions": {
                "transition_score": 0.20,
                "dialogue_ratio": 0.10,
            }
        }
        observation = ObservationAdapter(observation_data)

        intents = compiler.compile(observation, {})

        assert len(intents.intents) >= 2
        assert intents.intents[0].priority == IntentPriority.HIGH
        assert intents.intents[1].priority == IntentPriority.MEDIUM