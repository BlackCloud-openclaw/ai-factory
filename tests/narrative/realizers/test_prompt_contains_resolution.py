# tests/narrative/realizers/test_prompt_contains_resolution.py

import pytest
from uuid import uuid4

from src.narrative.context import (
    NarrativeContext,
    ChapterMetadata,
    StorySnapshot,
    ResolutionContext,
)
from src.narrative.intent import (
    NarrativeIntent,
    IntentSource,
    IntentDimension,
    IntentDirection,
    BuiltinDimensions,
)
from src.narrative.intent.conflict import Conflict, ConflictType
from src.narrative.conflict import ConflictResolution, ConflictStrategy
from src.narrative.realizers.prompts import build_editor_prompt


def test_prompt_contains_resolution():
    conflict = Conflict(
        type=ConflictType.DIRECTION_MISMATCH,
        intents=(),
        description="测试冲突",
    )
    resolution = ConflictResolution(
        conflict_id=conflict.id,
        strategy=ConflictStrategy.PRIORITY,
        rationale="优先选择增加对白",
        selected_intent=uuid4(),
    )
    res_ctx = ResolutionContext(
        conflicts=(conflict,),
        resolutions=(resolution,),
    )
    context = NarrativeContext(
        story=StorySnapshot(projection={}),
        metadata=ChapterMetadata(volume=1, chapter=1, scene_index=0, total_scenes=3),
        resolution_context=res_ctx,
    )

    intents = [
        NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension.dialogue(),
            desired_effect="编辑测试",
        )
    ]

    prompt = build_editor_prompt(
        artifact_text="测试文本",
        context=context,
        intents=intents,
        constraint_summary="保持事件一致",
        resolution_text="\n".join(r.to_prompt() for r in res_ctx.resolutions),
    )

    assert "冲突解决决策" in prompt
    assert "策略: priority" in prompt or "策略: PRIORITY" in prompt
    assert "优先选择增加对白" in prompt