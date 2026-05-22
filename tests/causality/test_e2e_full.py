#!/usr/bin/env python
"""
端到端集成测试 - 测试小说生成完整流程（大纲、规划、写作、验证、预算）
"""
import asyncio
import sys
import uuid
import json
sys.path.insert(0, '/home/data/projects/ai_factory')

from src.db import init_db_pool, get_db_pool
from src.orchestrator.state import AgentState
from src.orchestrator.graph import compile_workflow
from src.writing.event_store import NarrativeEventStore
from src.writing.events import ItemAcquireEvent
from src.agents.validator import ValidatorAgent
from src.writing.causality.budget import ConsistencyBudget


async def setup_test_novel(novel_id):
    pool = get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO novels (novel_id, title) VALUES ($1, $2) ON CONFLICT (novel_id) DO NOTHING",
            novel_id, "E2E Test Novel"
        )
        # 清理旧数据
        await conn.execute("DELETE FROM predicates WHERE novel_id = $1", novel_id)
        await conn.execute("DELETE FROM narrative_events WHERE novel_id = $1", novel_id)
        await conn.execute("DELETE FROM chapter_budget WHERE novel_id = $1", novel_id)
        await conn.execute("DELETE FROM scene_execution_units WHERE novel_id = $1", novel_id)


async def test_outline_generation():
    """测试大纲生成（通过工作流）"""
    novel_id = "e2e_outline_001"
    await setup_test_novel(novel_id)
    
    state = AgentState(
        user_input="生成一部修仙小说，共1卷，每卷3章，主角林逸，从炼气到筑基。",
        novel_id=novel_id,
        task_type="novel_outline",
        current_volume=1,
        current_chapter=1,
        current_scene_index=0
    )
    workflow = compile_workflow()
    result = await workflow.ainvoke(state.model_dump(), config={"recursion_limit": 10})
    outline = result.get("outline")
    assert outline is not None, "Outline generation failed"
    assert "volumes" in outline, "Outline missing volumes"
    assert len(outline["volumes"]) == 1, "Expected 1 volume"
    assert len(outline["volumes"][0]["chapters"]) == 3, "Expected 3 chapters"
    print("✅ Outline generation test passed")
    return outline


async def test_scene_plan_and_writing(novel_id, outline):
    """测试场景规划和写作（通过工作流）"""
    # 预置一些谓词（比如主角拥有神秘玉佩，用于因果校验）
    pool = get_db_pool()
    event = ItemAcquireEvent(
        event_id=str(uuid.uuid4()),
        actor="林逸",
        item="神秘玉佩",
        source="捡到"
    )
    store = NarrativeEventStore(pool)
    await store.append_event(novel_id, event, volume_num=1, chapter_num=1)
    
    # 初始化写作进度
    state = AgentState(
        user_input="继续写作",
        novel_id=novel_id,
        task_type="scene_plan",
        outline=outline,
        current_volume=1,
        current_chapter=1,
        current_scene_index=0,
        total_chapters_in_volume=3,
        resume=False
    )
    workflow = compile_workflow()
    # 限制步骤数（避免无限循环）
    result = await workflow.ainvoke(state.model_dump(), config={"recursion_limit": 50})
    
    # 检查是否生成了场景文件（至少第一章）
    import os
    from pathlib import Path
    chapter_file = Path(f"data/novels/{novel_id}/vol_001/chap_001.txt")
    assert chapter_file.exists(), f"Chapter file not found: {chapter_file}"
    content = chapter_file.read_text(encoding='utf-8')
    assert len(content) > 100, "Chapter content too short"
    print("✅ Scene planning and writing test passed")


async def test_consistency_budget():
    """测试一致性预算在 ValidatorAgent 中的集成"""
    novel_id = "e2e_budget_001"
    await setup_test_novel(novel_id)
    
    # 创建一个场景，故意触发对话规则（缺少 is_alive 谓词）
    scene_text = {
        "scene_text": "林逸开口说道：'你好。'",
        "events": [{"type": "dialogue", "actor": "林逸"}]
    }
    state = AgentState(
        novel_id=novel_id,
        current_volume=1,
        current_chapter=1,
        scene_text=json.dumps(scene_text),
        scene_plan={"must_events": []},
        task_type="scene_plan",
        validation_mode="novel"
    )
    validator = ValidatorAgent()
    
    # 第一次触发 warning，应通过
    result1 = await validator.run(state)
    assert result1["validation_result"]["passed"] is True, "First warning should pass"
    
    # 第二次仍应通过
    result2 = await validator.run(state)
    assert result2["validation_result"]["passed"] is True
    
    # 第三次预算耗尽，应升级为 error，导致 passed=False
    result3 = await validator.run(state)
    assert result3["validation_result"]["passed"] is False, "Third warning should upgrade to error"
    print("✅ Consistency budget test passed")


async def main():
    print("Initializing DB...")
    await init_db_pool()
    
    print("\n--- Testing Outline Generation ---")
    outline = await test_outline_generation()
    
    print("\n--- Testing Scene Plan and Writing ---")
    novel_id = "e2e_full_001"
    await setup_test_novel(novel_id)
    await test_scene_plan_and_writing(novel_id, outline)
    
    print("\n--- Testing Consistency Budget Integration ---")
    await test_consistency_budget()
    
    print("\n🎉 All E2E tests passed!")


if __name__ == "__main__":
    asyncio.run(main())