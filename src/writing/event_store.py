import json
import asyncpg
from typing import List, Dict, Any
from .events import Event

class EventStore:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def insert_event(self, event: Event) -> int:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO events (event_id, type, payload, novel_id, chapter_id)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING sequence_id
                """,
                event.event_id,
                event.type,
                json.dumps(event.payload),
                event.novel_id,
                event.chapter_id
            )
            return row["sequence_id"]

    async def load_events(self, novel_id: str, from_sequence: int = 0) -> List[Event]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT sequence_id, event_id, type, payload, created_at, novel_id, chapter_id
                FROM events
                WHERE novel_id = $1 AND sequence_id > $2
                ORDER BY sequence_id ASC
                """,
                novel_id, from_sequence
            )
            events = []
            for r in rows:
                events.append(Event(
                    event_id=str(r["event_id"]),               # UUID -> str
                    sequence_id=r["sequence_id"],
                    type=r["type"],
                    payload=json.loads(r["payload"]),          # JSON string -> dict
                    created_at=r["created_at"],
                    novel_id=r["novel_id"],
                    chapter_id=r["chapter_id"]
                ))
            return events

    async def save_snapshot(self, novel_id: str, state: Dict, last_sequence_id: int):
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO novels (novel_id, current_state, last_sequence_id, updated_at)
                VALUES ($1, $2, $3, NOW())
                ON CONFLICT (novel_id) DO UPDATE
                SET current_state = EXCLUDED.current_state,
                    last_sequence_id = EXCLUDED.last_sequence_id,
                    updated_at = NOW()
                """,
                novel_id,
                json.dumps(state),
                last_sequence_id
            )

    async def load_snapshot(self, novel_id: str) -> tuple[Dict, int]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT current_state, last_sequence_id FROM novels WHERE novel_id = $1",
                novel_id
            )
            if row:
                state = row["current_state"]
                if isinstance(state, str):
                    import json
                    state = json.loads(state)
                return state or {}, row["last_sequence_id"] or 0
            return {}, 0