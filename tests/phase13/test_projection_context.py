"""
Phase 13.2: ProjectionContext 测试
"""

import pytest
from src.writing.narrative_projection import NarrativeProjection
from src.writing.narrative_intent import SceneRole
from src.writing.projection_context import ProjectionContext


def test_projection_context_from_projection():
    """测试从 Projection 构建 Context"""
    proj = NarrativeProjection(
        projection_id="test",
        chapter_id="ch1",
        active_conflict="师门秘密",
        unresolved_threads=["长老身份", "血脉来源"],
        last_intent_id="intent",
        last_scene_role=SceneRole.SETUP,
        version=1,
    )
    ctx = ProjectionContext.from_projection(proj)
    assert ctx.active_conflict == "师门秘密"
    assert len(ctx.unresolved_threads) == 2
    assert "血脉来源" in ctx.unresolved_threads
    assert ctx.next_pressure is None


def test_projection_context_limits_threads():
    """验证 Context 限制线程数量"""
    proj = NarrativeProjection(
        projection_id="test",
        chapter_id="ch1",
        active_conflict="冲突",
        unresolved_threads=[f"thread_{i}" for i in range(10)],
        last_intent_id="intent",
        last_scene_role=SceneRole.SETUP,
        version=1,
    )
    ctx = ProjectionContext.from_projection(proj)
    assert len(ctx.unresolved_threads) == 5


def test_projection_context_to_prompt():
    """验证 Prompt 文本生成"""
    ctx = ProjectionContext(
        active_conflict="师门秘密",
        unresolved_threads=["长老身份", "血脉来源"],
        next_pressure="必须做出选择"
    )
    text = ctx.to_prompt_text()
    assert "Active Conflict: 师门秘密" in text
    assert "- 长老身份" in text
    assert "- 血脉来源" in text
    assert "Next Pressure: 必须做出选择" in text
    assert "你的 NarrativeIntent 必须回应这些未完成的叙事线" in text