from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import asyncpg
from uuid import UUID, uuid4

@dataclass
class Loop:
    id: UUID
    novel_id: str
    title: str
    description: str
    status: str  # 'active' | 'resolved' | 'abandoned'
    progress: float
    owner: Optional[str]
    priority: int
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime] = None

class LoopStore:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def get_active_loop(self, novel_id: str) -> Optional[Loop]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM loop_store WHERE novel_id=$1 AND status='active' ORDER BY priority DESC, created_at LIMIT 1",
                novel_id
            )
        return self._row_to_loop(row) if row else None

    async def create_loop(self, novel_id: str, title: str, description: str, owner: str = None) -> Loop:
        loop_id = uuid4()
        async with self.pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO loop_store (id, novel_id, title, description, owner_character_id, status, progress)
                   VALUES ($1, $2, $3, $4, $5, 'active', 0.0)""",
                loop_id, novel_id, title, description, owner
            )
        return await self.get_loop(loop_id)

    async def update_progress(self, loop_id: UUID, new_progress: float):
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE loop_store SET progress=$1, updated_at=now() WHERE id=$2",
                min(1.0, max(0.0, new_progress)), loop_id
            )

    async def resolve_loop(self, loop_id: UUID):
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE loop_store SET status='resolved', resolved_at=now() WHERE id=$1",
                loop_id
            )

    async def get_loop(self, loop_id: UUID) -> Optional[Loop]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM loop_store WHERE id=$1", loop_id)
        return self._row_to_loop(row) if row else None

    @staticmethod
    def _row_to_loop(row) -> Loop:
        return Loop(
            id=row["id"],
            novel_id=row["novel_id"],
            title=row["title"],
            description=row["description"],
            status=row["status"],
            progress=row["progress"],
            owner=row["owner_character_id"],
            priority=row["priority"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            resolved_at=row["resolved_at"]
        )