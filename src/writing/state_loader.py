import json
from typing import Optional, Tuple
from src.db import get_db_pool
from src.writing.world_state import WorldState
from src.writing.delta import StateDelta
from src.writing.event_store import NarrativeEventStore
from src.writing.snapshot import SnapshotManager

async def load_state_at_event(novel_id: str, target_event_id: int) -> Tuple[Optional[WorldState], int]:
    pool = get_db_pool()
    snap_mgr = SnapshotManager(pool)
    event_store = NarrativeEventStore(pool)

    async with pool.acquire() as conn:
        snap_row = await conn.fetchrow(
            """
            SELECT snapshot_id, last_event_id, world_state
            FROM world_snapshots
            WHERE novel_id = $1 AND last_event_id <= $2
            ORDER BY last_event_id DESC
            LIMIT 1
            """,
            novel_id, target_event_id
        )
    if snap_row:
        world = WorldState.from_dict(json.loads(snap_row["world_state"]))
        last_event = snap_row["last_event_id"]
    else:
        world = WorldState()
        last_event = 0

    # 分页加载事件
    last_loaded = last_event
    while True:
        batch = await event_store.get_events_since(novel_id, since_event_id=last_loaded, limit=5000)
        if not batch:
            break
        for evt_id, evt in batch:
            if evt_id > target_event_id:
                break
            delta = StateDelta(events=[evt])
            world = delta.apply_to(world)
            last_loaded = evt_id
        else:
            continue
        break
    return world, last_event

async def load_state_at_chapter(novel_id: str, volume_num: int, chapter_num: int) -> Tuple[Optional[WorldState], int]:
    pool = get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT MAX(id) as last_id
            FROM narrative_events
            WHERE novel_id = $1 AND volume_num = $2 AND chapter_num = $3
            """,
            novel_id, volume_num, chapter_num
        )
        last_id = row["last_id"] if row else 0
    if last_id == 0:
        return None, 0
    return await load_state_at_event(novel_id, last_id)
