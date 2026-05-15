import asyncpg
from typing import Optional, Tuple
from datetime import datetime
from .world_state import WorldState
from .memory_hierarchy import CompressedState  # 确保该模块存在


class SnapshotManager:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool
    
    async def save_snapshot(
        self,
        novel_id: str,
        world_state: WorldState,
        last_event_id: int,
        volume_num: int = None,
        chapter_num: int = None,
        compressed_state: Optional[CompressedState] = None,
    ) -> int:
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "SELECT COALESCE(MAX(snapshot_id), 0) + 1 as next_id FROM world_snapshots WHERE novel_id = $1",
                    novel_id
                )
                snap_id = row["next_id"]
                await conn.execute("""
                    INSERT INTO world_snapshots
                    (novel_id, snapshot_id, volume_num, chapter_num, last_event_id, world_state, compressed_state, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, novel_id, snap_id, volume_num, chapter_num, last_event_id,
                   world_state.model_dump_json(),
                   compressed_state.model_dump_json() if compressed_state else None,
                   datetime.now())
                return snap_id
    
    async def load_latest_snapshot(self, novel_id: str) -> Tuple[Optional[WorldState], Optional[CompressedState], int]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT world_state, compressed_state, last_event_id
                FROM world_snapshots
                WHERE novel_id = $1
                ORDER BY snapshot_id DESC
                LIMIT 1
            """, novel_id)
            if not row:
                return None, None, 0
            world = WorldState.model_validate_json(row["world_state"])
            comp = None
            if row["compressed_state"]:
                comp = CompressedState.model_validate_json(row["compressed_state"])
            return world, comp, row["last_event_id"]
    
    async def prune_snapshots(self, novel_id: str, keep: int = 3):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                DELETE FROM world_snapshots
                WHERE novel_id = $1 AND snapshot_id NOT IN (
                    SELECT snapshot_id FROM world_snapshots
                    WHERE novel_id = $1
                    ORDER BY snapshot_id DESC
                    LIMIT $2
                )
            """, novel_id, keep)
    
    async def should_snapshot(self, novel_id: str, current_event_id: int, interval_events: int = 1000) -> bool:
        last_event_id = await self._get_last_snapshot_event_id(novel_id)
        return current_event_id - last_event_id >= interval_events
    
    async def get_last_snapshot_event_id(self, novel_id: str) -> int:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT last_event_id FROM world_snapshots
                WHERE novel_id = $1 ORDER BY snapshot_id DESC LIMIT 1
            """, novel_id)
            return row["last_event_id"] if row else 0