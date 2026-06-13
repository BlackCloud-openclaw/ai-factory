#!/usr/bin/env python
"""
系统状态报告脚本 - 查询数据库中的叙事熵、快照、投影健康等指标
用法: python scripts/report.py
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.db import init_db_pool, close_db_pool, get_db_pool


async def get_event_count(pool) -> int:
    async with pool.acquire() as conn:
        row = await conn.fetchval(
            "SELECT COUNT(*) FROM narrative_events WHERE novel_id = 'simple_long_novel_001'"
        )
        return row or 0


async def get_snapshot_info(pool):
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT snapshot_id, volume_num, chapter_num, last_event_id,
                   pg_column_size(world_state) as world_state_size,
                   compressed_state IS NOT NULL as has_compressed,
                   created_at
            FROM world_snapshots
            WHERE novel_id = 'simple_long_novel_001'
            ORDER BY snapshot_id DESC
            LIMIT 5
            """
        )
        return [dict(row) for row in rows]


async def get_entropy_history(pool):
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT snapshot_id, volume_num, chapter_num,
                   compressed_state->>'narrative_entropy' as narrative_entropy,
                   compressed_state->>'local_entropy' as local_entropy,
                   compressed_state->>'arc_entropy' as arc_entropy,
                   compressed_state->>'civilization_entropy' as civilization_entropy,
                   compressed_state->>'entropy_history' as entropy_history
            FROM world_snapshots
            WHERE novel_id = 'simple_long_novel_001'
              AND compressed_state IS NOT NULL
            ORDER BY snapshot_id ASC
            """
        )
        result = []
        for row in rows:
            item = {
                "snapshot_id": row["snapshot_id"],
                "volume": row["volume_num"],
                "chapter": row["chapter_num"],
                "narrative_entropy": float(row["narrative_entropy"]) if row["narrative_entropy"] else None,
                "local_entropy": float(row["local_entropy"]) if row["local_entropy"] else None,
                "arc_entropy": float(row["arc_entropy"]) if row["arc_entropy"] else None,
                "civilization_entropy": float(row["civilization_entropy"]) if row["civilization_entropy"] else None,
            }
            result.append(item)
        return result


async def get_projection_health(pool):
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT last_projected_event_id, projection_lag_events,
                   drift_level, validator_mode, updated_at
            FROM projection_health
            WHERE novel_id = 'simple_long_novel_001'
            """
        )
        return dict(row) if row else None


async def get_dead_letter_count(pool):
    async with pool.acquire() as conn:
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM projection_dead_letters WHERE novel_id = 'simple_long_novel_001'"
        )
        return count or 0


async def get_latest_event_id(pool):
    async with pool.acquire() as conn:
        row = await conn.fetchval(
            "SELECT MAX(id) FROM narrative_events WHERE novel_id = 'simple_long_novel_001'"
        )
        return row or 0


async def get_snapshot_last_event(pool):
    async with pool.acquire() as conn:
        row = await conn.fetchval(
            """
            SELECT last_event_id FROM world_snapshots
            WHERE novel_id = 'simple_long_novel_001'
            ORDER BY snapshot_id DESC LIMIT 1
            """
        )
        return row or 0


async def main():
    await init_db_pool()
    pool = get_db_pool()
    if not pool:
        print("❌ 无法连接到数据库")
        return

    from datetime import datetime
    print("=" * 80)
    print("AI Factory 系统状态报告")
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 1. 事件统计
    total_events = await get_event_count(pool)
    latest_event_id = await get_latest_event_id(pool)
    snapshot_last_event = await get_snapshot_last_event(pool)
    print(f"\n📊 事件统计")
    print(f"  总事件数: {total_events}")
    print(f"  最新事件 ID: {latest_event_id}")
    print(f"  最新快照包含的事件 ID: {snapshot_last_event}")
    if latest_event_id > snapshot_last_event:
        print(f"  ⚠️ 快照后还有 {latest_event_id - snapshot_last_event} 个事件未包含在快照中")
    else:
        print(f"  ✅ 快照已包含所有事件")

    # 2. 快照信息
    snapshots = await get_snapshot_info(pool)
    print(f"\n📸 最近5个快照")
    for snap in snapshots:
        print(f"  ID {snap['snapshot_id']}: 卷{snap['volume_num']}章{snap['chapter_num']} | "
              f"事件ID {snap['last_event_id']} | 大小 {snap['world_state_size']} 字节 | "
              f"压缩状态: {'有' if snap['has_compressed'] else '无'}")

    # 3. 熵值历史
    entropy_history = await get_entropy_history(pool)
    print(f"\n📈 熵值历史 (按卷章排序)")
    print(f"  {'卷':<4} {'章':<4} {'叙事熵':<10} {'局部熵':<10} {'弧线熵':<10} {'文明熵':<10}")
    for e in entropy_history:
        print(f"  {e['volume']:<4} {e['chapter']:<4} "
              f"{e['narrative_entropy'] or 0:<10.3f} "
              f"{e['local_entropy'] or 0:<10.3f} "
              f"{e['arc_entropy'] or 0:<10.3f} "
              f"{e['civilization_entropy'] or 0:<10.3f}")

    # 4. 投影健康
    health = await get_projection_health(pool)
    print(f"\n🔧 投影健康状态")
    if health:
        print(f"  最后投影事件ID: {health.get('last_projected_event_id')}")
        print(f"  投影滞后事件数: {health.get('projection_lag_events')}")
        print(f"  漂移级别: {health.get('drift_level')}")
        print(f"  验证器模式: {health.get('validator_mode')}")
        print(f"  最后更新: {health.get('updated_at')}")
    else:
        print("  无投影健康记录")

    # 5. 死信队列
    dead_letter_count = await get_dead_letter_count(pool)
    print(f"\n⚠️ 投影死信队列")
    print(f"  死信数量: {dead_letter_count}")

    # 6. 综合结论
    print(f"\n✅ 综合结论")
    if latest_event_id == snapshot_last_event and dead_letter_count == 0 and (health and health.get('drift_level') == 'INFO'):
        print("  系统状态良好，无异常")
    else:
        if latest_event_id > snapshot_last_event:
            print("  ⚠️ 快照落后于事件，建议重新生成快照")
        if dead_letter_count > 0:
            print(f"  ⚠️ 存在 {dead_letter_count} 个死信事件，请检查 projection_dead_letters 表")
        if health and health.get('drift_level') not in ('INFO', None):
            print(f"  ⚠️ 投影漂移级别 {health.get('drift_level')}，可能需要重建投影")

    await close_db_pool()


if __name__ == "__main__":
    asyncio.run(main())