#!/usr/bin/env python
"""状态审计重放脚本：从事件流完全重建状态，与最后一次审计哈希对比"""

import sys
import os
# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from src.db import init_db_pool, close_db_pool, get_db_pool
from src.writing.event_store import NarrativeEventStore
from src.writing.world_state import WorldState
from src.writing.delta import StateDelta
from src.common.canonical import canonical_hash


async def rebuild_state_from_events(novel_id: str, target_event_id: int = None):
    """从事件流重建世界状态（不依赖快照）"""
    pool = get_db_pool()
    event_store = NarrativeEventStore(pool)

    world = WorldState()
    events = await event_store.get_events_since(novel_id, since_event_id=0, limit=100000)
    for evt_id, evt in events:
        if target_event_id is not None and evt_id > target_event_id:
            break
        delta = StateDelta(events=[evt])
        world = delta.apply_to(world)
    return world


async def audit_replay(novel_id: str):
    await init_db_pool()
    pool = get_db_pool()

    # 1. 获取最后一次审计记录
    async with pool.acquire() as conn:
        last_audit = await conn.fetchrow(
            """
            SELECT state_hash, last_event_id, node_name, step_count
            FROM state_audit
            WHERE novel_id = $1
            ORDER BY created_at DESC
            LIMIT 1
            """,
            novel_id
        )
        if not last_audit:
            print(f"No audit records found for novel {novel_id}")
            await close_db_pool()
            return

    expected_hash = last_audit["state_hash"]
    last_event_id = last_audit["last_event_id"] or 0
    node_name = last_audit["node_name"]
    step_count = last_audit["step_count"]

    print(f"Auditing novel {novel_id} at node {node_name}, step {step_count}, last_event_id={last_event_id}")

    # 2. 从事件流重建世界状态（到 last_event_id）
    world = await rebuild_state_from_events(novel_id, target_event_id=last_event_id)

    # 3. 从 novels 表加载其他元数据
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT outline, current_volume, current_chapter, current_scene_index FROM novels WHERE novel_id = $1",
            novel_id
        )
        if row:
            outline_hash = canonical_hash(row["outline"]) if row["outline"] else ""
            meta = {
                "current_volume": row["current_volume"],
                "current_chapter": row["current_chapter"],
                "current_scene_index": row["current_scene_index"],
                "outline_hash": outline_hash,
            }
            meta_hash = canonical_hash(meta)
        else:
            meta_hash = ""

    world_hash = canonical_hash(world.to_dict())
    combined_hash = canonical_hash({"world": world_hash, "meta": meta_hash})
    
    if combined_hash == expected_hash:
        print(f"✅ Audit passed for {novel_id}")
    else:
        print(f"❌ Audit failed for {novel_id}")
        print(f"   Expected: {expected_hash}")
        print(f"   Actual:   {combined_hash}")
        print(f"   World hash: {world_hash}")
        print(f"   Meta hash:  {meta_hash}")
    
    await close_db_pool()


if __name__ == "__main__":
    novel_id = sys.argv[1] if len(sys.argv) > 1 else "simple_long_novel_001"
    asyncio.run(audit_replay(novel_id))