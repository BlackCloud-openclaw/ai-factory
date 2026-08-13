# tests/narrative/resolution/test_context_builder.py

import pytest
from uuid import uuid4

from src.narrative.resolution.context_builder import build_resolution_context, enrich_narrative_context
from src.narrative.intent import ResolutionPlan
from src.narrative.context import NarrativeContext, ChapterMetadata, StorySnapshot
from src.narrative.conflict import ConflictResolution, ConflictStrategy


def test_build_context_with_resolutions():
    res = ConflictResolution(
        conflict_id=uuid4(),
        strategy=ConflictStrategy.PRIORITY,
        rationale="测试决议",
    )
    plan = ResolutionPlan(resolutions=(res,))
    ctx = build_resolution_context(plan)
    assert ctx is not None
    assert len(ctx.resolutions) == 1
    assert ctx.resolutions[0] == res


def test_build_context_no_resolutions():
    plan = ResolutionPlan()
    ctx = build_resolution_context(plan)
    assert ctx is None


def test_enrich_context():
    base = NarrativeContext(
        story=StorySnapshot(projection={}),
        metadata=ChapterMetadata(volume=1, chapter=1, scene_index=0, total_scenes=3)
    )
    res = ConflictResolution(
        conflict_id=uuid4(),
        strategy=ConflictStrategy.PRIORITY,
        rationale="测试",
    )
    plan = ResolutionPlan(resolutions=(res,))
    enriched = enrich_narrative_context(base, plan)
    assert enriched.resolution_context is not None
    assert len(enriched.resolution_context.resolutions) == 1
    assert enriched.resolution_context.resolutions[0] == res


def test_enrich_context_no_resolution():
    base = NarrativeContext(
        story=StorySnapshot(projection={}),
        metadata=ChapterMetadata(volume=1, chapter=1, scene_index=0, total_scenes=3)
    )
    plan = ResolutionPlan()
    enriched = enrich_narrative_context(base, plan)
    assert enriched is base
    assert enriched.resolution_context is None