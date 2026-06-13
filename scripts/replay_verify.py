#!/usr/bin/env python
"""
状态重放验证脚本 - 确保事件溯源状态与快照一致
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db import init_db_pool, close_db_pool, get_db_pool
from src.writing.event_store import NarrativeEventStore
from src.writing.snapshot import SnapshotManager
from src.writing.world_state import WorldState
from src.writing.delta import StateDelta
from src.common.canonical import canonical_hash


async def verify_replay(novel_id: str):
    """验证从事件流重放的状态与快照一致"""
    await init_db_pool()
    pool = get_db_pool()
    event_store = NarrativeEventStore(pool)
    snap_mgr = SnapshotManager(pool)
    
    # 加载最新快照
    world_from_snapshot, compressed, last_event_id = await snap_mgr.load_latest_snapshot(novel_id)
    if world_from_snapshot is None:
        print(f"No snapshot found for {novel_id}")
        await close_db_pool()
        return False
    
    print(f"Loaded snapshot with last_event_id={last_event_id}")
    
    # 从事件流重建状态（全量）
    world_rebuilt = WorldState()
    events_with_id = await event_store.get_events_since(novel_id, since_event_id=0, limit=100000)
    for evt_id, evt in events_with_id:
        delta = StateDelta(events=[evt])
        world_rebuilt = delta.apply_to(world_rebuilt)
    
    print(f"Rebuilt state from {len(events_with_id)} events")
    
    # 计算哈希
    hash_snapshot = canonical_hash(world_from_snapshot.model_dump())
    hash_rebuilt = canonical_hash(world_rebuilt.model_dump())
    
    if hash_snapshot == hash_rebuilt:
        print("✅ State verification PASSED: snapshot matches replay")
        return True
    else:
        print(f"❌ State verification FAILED: hash mismatch")
        print(f"   Snapshot hash: {hash_snapshot}")
        print(f"   Rebuilt hash:  {hash_rebuilt}")
        return False


if __name__ == "__main__":
    novel_id = sys.argv[1] if len(sys.argv) > 1 else "simple_long_novel_001"
    result = asyncio.run(verify_replay(novel_id))
    sys.exit(0 if result else 1)