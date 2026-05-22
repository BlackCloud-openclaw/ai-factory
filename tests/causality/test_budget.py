#!/usr/bin/env python
import asyncio
import sys
sys.path.insert(0, '/home/data/projects/ai_factory')

from src.db import init_db_pool, get_db_pool
from src.writing.causality.budget import ConsistencyBudget


async def test_budget_consume():
    """测试预算消费逻辑（不依赖 ValidatorAgent）"""
    novel_id = "test_budget_001"
    # 初始化数据库连接
    await init_db_pool()
    pool = get_db_pool()
    if not pool:
        print("❌ Database pool not initialized")
        return

    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO novels (novel_id, title) VALUES ($1, $2) ON CONFLICT (novel_id) DO NOTHING",
            novel_id, "Test Novel"
        )
        await conn.execute("DELETE FROM chapter_budget WHERE novel_id = $1", novel_id)

    budget = ConsistencyBudget(novel_id, 1, 1)
    # 前两次 warning 应返回 True
    assert await budget.consume("warning") is True
    await budget.load()
    assert budget.remaining_warnings == 2
    assert await budget.consume("warning") is True
    await budget.load()
    assert budget.remaining_warnings == 1
    # 第三次 warning 应返回 False（预算耗尽）
    assert await budget.consume("warning") is False
    await budget.load()
    assert budget.remaining_warnings == 0
    # 第四次仍为 False
    assert await budget.consume("warning") is False

    # 重置预算
    await budget.reset()
    assert budget.remaining_warnings == 3

    # soft_contradiction 测试
    assert await budget.consume("soft_contradiction") is True
    assert budget.remaining_soft == 0
    assert await budget.consume("soft_contradiction") is False

    print("✅ Budget consume test passed")


if __name__ == "__main__":
    asyncio.run(test_budget_consume())