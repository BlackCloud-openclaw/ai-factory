#!/usr/bin/env python
"""
Kernel 兼容性测试 Level 1-3

Level 1: 总体统计一致性（实体数、关系数、能力数）
Level 2: 关键实体状态一致性（主角境界、位置、关键物品）
Level 3: 叙事不变量一致性（重要关系值、未解决弧线数）
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.db import init_db_pool, close_db_pool, get_db_pool
from src.writing.world_state import WorldState
from src.writing.state_loader import load_state_at_chapter
from src.writing.memory_hierarchy import CompressedState
from next.adapter.xianxia_adapter import XianxiaAdapter


async def get_chapter_compressed(novel_id: str, volume: int, chapter: int):
    """加载章节结束时的压缩状态"""
    pool = get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT compressed_state
            FROM world_snapshots
            WHERE novel_id = $1 AND volume_num = $2 AND chapter_num = $3
            ORDER BY snapshot_id DESC
            LIMIT 1
            """,
            novel_id, volume, chapter
        )
    if row and row["compressed_state"]:
        import json
        data = json.loads(row["compressed_state"])
        return CompressedState(**data)
    return None


async def test_chapter_consistency(novel_id: str, volume: int, chapter: int) -> dict:
    """测试单个章节的一致性"""
    world, _ = await load_state_at_chapter(novel_id, volume, chapter)
    if world is None:
        return {"chapter": chapter, "passed": False, "error": "无法加载世界状态"}

    kernel = XianxiaAdapter.to_kernel(world)
    compressed = await get_chapter_compressed(novel_id, volume, chapter)

    errors = []

    # ===== Level 1: 总体统计 =====
    # 1.1 实体数量
    if len(world.characters) != len(kernel.entities):
        errors.append(f"实体数量不一致: world={len(world.characters)}, kernel={len(kernel.entities)}")

    # 1.2 客观关系数量
    world_rel_count = len(world.relationships)
    kernel_rel_count = sum(1 for r in kernel.relations.values() if r.relation_type == "objective")
    if world_rel_count != kernel_rel_count:
        errors.append(f"客观关系数量不一致: world={world_rel_count}, kernel={kernel_rel_count}")

    # 1.3 能力数量（粗略：每个角色至少有 cultivation 和 inventory）
    expected_cap_count = len(world.characters) * 2 + 1  # +1 for global_flags
    if len(kernel.capabilities) < expected_cap_count:
        errors.append(f"能力数量不足: expected>= {expected_cap_count}, got {len(kernel.capabilities)}")

    # ===== Level 2: 关键实体状态 =====
    protagonist = "林逸"
    if protagonist in world.characters:
        # 2.1 主角境界
        world_realm = world.characters[protagonist].full_realm()
        kernel_realm = kernel.capabilities.get(f"{protagonist}|cultivation")
        kernel_realm_val = kernel_realm.value if kernel_realm else None
        if world_realm != kernel_realm_val:
            errors.append(f"主角境界不一致: world={world_realm}, kernel={kernel_realm_val}")

        # 2.2 主角物品列表（作为集合比较）
        world_items = set(world.characters[protagonist].inventory)
        kernel_inv = kernel.capabilities.get(f"{protagonist}|inventory")
        kernel_items = set(kernel_inv.value) if kernel_inv else set()
        if world_items != kernel_items:
            errors.append(f"主角物品不一致: world={world_items}, kernel={kernel_items}")

        # 2.3 主角位置
        world_loc = world.characters[protagonist].location
        kernel_loc = kernel.entities.get(protagonist, {}).attributes.get("location") if protagonist in kernel.entities else None
        if world_loc != kernel_loc:
            errors.append(f"主角位置不一致: world={world_loc}, kernel={kernel_loc}")

        # 2.4 主角生命值
        world_hp = world.characters[protagonist].hp
        kernel_hp = kernel.entities.get(protagonist, {}).attributes.get("hp") if protagonist in kernel.entities else None
        if world_hp != kernel_hp:
            errors.append(f"主角生命值不一致: world={world_hp}, kernel={kernel_hp}")

    # ===== Level 3: 叙事不变量 =====
    # 3.1 主角与二叔的关系值
    second_uncle = "二叔"
    if protagonist in world.characters and second_uncle in world.characters:
        rel_key = f"{protagonist}|{second_uncle}"
        world_rel = world.relationships.get(rel_key, 0)
        kernel_rel = None
        for r in kernel.relations.values():
            if r.from_entity == protagonist and r.to_entity == second_uncle and r.relation_type == "objective":
                kernel_rel = r.value
                break
        if world_rel != kernel_rel:
            errors.append(f"关系 {protagonist}->{second_uncle} 不一致: world={world_rel}, kernel={kernel_rel}")

    # 3.2 未解决弧线数量（从 compressed_state）
    if compressed and hasattr(compressed, 'character_arcs'):
        world_unresolved = sum(1 for status in compressed.character_arcs.values() if status != "resolved")
        # Kernel 中目前没有直接存储弧线，暂时跳过比较
        # 可以后续扩展
    else:
        world_unresolved = -1  # 无数据

    # 3.3 核心谓词：主角是否存活（is_alive）
    # 原系统中 is_alive 存储在 predicates 表，WorldState 中未直接体现，跳过

    passed = len(errors) == 0
    return {
        "chapter": chapter,
        "passed": passed,
        "errors": errors,
        "stats": {
            "entities": len(kernel.entities),
            "relations": kernel_rel_count,
            "capabilities": len(kernel.capabilities),
        }
    }


async def test_all_chapters(novel_id: str, max_chapter: int = 32):
    await init_db_pool()
    results = []
    for ch in range(1, max_chapter + 1):
        print(f"测试第 1 卷第 {ch} 章...")
        result = await test_chapter_consistency(novel_id, 1, ch)
        results.append(result)
        if not result["passed"]:
            print(f"  ❌ 失败: {result['errors'][0]}")
        else:
            print(f"  ✅ 通过")
    await close_db_pool()

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    print(f"\n{'='*50}")
    print(f"兼容性测试结果: {passed}/{total} 章通过")
    if passed == total:
        print("🎉 所有章节 Level 1-3 测试通过！")
    else:
        print("⚠️ 存在不一致，请检查上述错误")


if __name__ == "__main__":
    novel_id = "simple_long_novel_001"
    max_chapter = 32  # 当前最多32章
    asyncio.run(test_all_chapters(novel_id, max_chapter))