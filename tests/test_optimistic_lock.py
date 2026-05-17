#!/usr/bin/env python
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from src.db import init_db_pool, close_db_pool
from src.orchestrator.state import AgentState
from src.orchestrator.nodes import save_memory_node

async def test_concurrent_save():
    await init_db_pool()
    
    novel_id = "test_optimistic_lock"
    
    # 先插入一条初始记录
    from src.db import get_db_pool
    pool = get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO novels (novel_id, title, outline, current_volume, current_chapter, current_scene_index, revision)
            VALUES ($1, $2, $3, $4, $5, $6, 0)
            ON CONFLICT (novel_id) DO NOTHING
        """, novel_id, "测试小说", "{}", 1, 1, 0)
    
    # 创建两个并发的 state，各自修改 outline
    state1 = AgentState(
        novel_id=novel_id,
        outline={"version": "A", "data": "修改1"},
        current_volume=1,
        current_chapter=1,
        current_scene_index=0
    )
    state2 = AgentState(
        novel_id=novel_id,
        outline={"version": "B", "data": "修改2"},
        current_volume=1,
        current_chapter=1,
        current_scene_index=0
    )
    
    # 并发执行保存
    results = await asyncio.gather(
        save_memory_node(state1),
        save_memory_node(state2),
        return_exceptions=True
    )
    
    print("Results:", results)
    
    # 检查最终 revision 值
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT revision, outline FROM novels WHERE novel_id = $1", novel_id)
        print(f"Final revision: {row['revision']}, outline: {row['outline']}")
    
    await close_db_pool()

if __name__ == "__main__":
    asyncio.run(test_concurrent_save())