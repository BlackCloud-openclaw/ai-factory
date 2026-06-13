#!/usr/bin/env python
"""
生成确定性重放测试的黄金数据集

使用方法：
    python scripts/generate_golden.py --novel-id simple_long_novel_001 --output tests/golden_replay/golden.json
"""

import asyncio
import json
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db import init_db_pool, close_db_pool, get_db_pool
from src.writing.event_store import NarrativeEventStore
from src.writing.snapshot import SnapshotManager
from src.writing.world_state import WorldState
from src.writing.delta import StateDelta
from src.common.canonical import canonical_hash


async def generate_golden(novel_id: str, output_path: Path):
    """生成黄金数据集"""
    await init_db_pool()
    pool = get_db_pool()
    event_store = NarrativeEventStore(pool)
    snap_mgr = SnapshotManager(pool)

    # 获取所有章节的 checkpoint（每章结束后）
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT volume_num, chapter_num
            FROM narrative_events
            WHERE novel_id = $1
            ORDER BY volume_num, chapter_num
            """,
            novel_id
        )

    checkpoints = []
    for vol, chap in rows:
        # 获取该章节最后一个事件的 ID
        async with pool.acquire() as conn2:
            last_event_row = await conn2.fetchrow(
                """
                SELECT MAX(id) as last_id
                FROM narrative_events
                WHERE novel_id = $1 AND volume_num = $2 AND chapter_num = $3
                """,
                novel_id, vol, chap
            )
        if not last_event_row or not last_event_row["last_id"]:
            continue
        event_id = last_event_row["last_id"]

        # 重建到该事件的状态
        world, _, _ = await snap_mgr.load_latest_snapshot(novel_id)
        if world is None:
            world = WorldState()
            last_projected = 0
        else:
            last_projected = await snap_mgr.get_last_snapshot_event_id(novel_id)

        events = await event_store.get_events_since(novel_id, last_projected, limit=10000)
        for evt_id, evt in events:
            if evt_id > event_id:
                break
            world = world.apply_delta(StateDelta(events=[evt]))

        # 计算哈希
        world_hash = world.get_state_hash()

        # 获取当前活跃谓词的哈希（可选）
        active_preds = await event_store._load_active_predicates(novel_id)
        preds_hash = canonical_hash({k: v.object for k, v in active_preds.items()})

        checkpoints.append({
            "volume": vol,
            "chapter": chap,
            "event_id": event_id,
            "world_hash": world_hash,
            "predicate_hash": preds_hash,
        })

    golden = {
        "novel_id": novel_id,
        "generated_at": asyncio.get_event_loop().time(),
        "checkpoints": checkpoints,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(golden, f, indent=2)

    print(f"✅ Golden dataset saved to {output_path}")
    await close_db_pool()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--novel-id", default="simple_long_novel_001")
    parser.add_argument("--output", default="tests/golden_replay/golden.json")
    args = parser.parse_args()
    asyncio.run(generate_golden(args.novel_id, Path(args.output)))