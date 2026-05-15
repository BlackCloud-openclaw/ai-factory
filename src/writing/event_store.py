"""
事件存储 - 持久化叙事事件

特性：
- 幂等写入（基于 event_uuid）
- 版本化存储
- 支持按 novel_id/chapter 查询
- 自动 upcast 历史事件
"""
import asyncpg
import json
from typing import List, Optional, Tuple
from datetime import datetime

from .events import NarrativeEvent, event_to_dict, event_from_dict
from .event_upcaster import EventUpcaster
from src.common.logging import setup_logging

logger = setup_logging("writing.event_store")


class NarrativeEventStore:
    """叙事事件存储（新架构核心）"""

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def append_event(
        self,
        novel_id: str,
        event: NarrativeEvent,
        volume_num: int = None,
        chapter_num: int = None,
        scene_id: int = None,
    ) -> str:
        """存储单个事件（幂等）"""
        event_data = event_to_dict(event)
        event_data = EventUpcaster.upcast(event_data)

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO narrative_events 
                (event_uuid, novel_id, volume_num, chapter_num, scene_id, 
                 event_type, event_data, event_version, timestamp)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (event_uuid) DO NOTHING
                """,
                event.event_id,
                novel_id,
                volume_num,
                chapter_num,
                scene_id,
                event.type.value,
                json.dumps(event_data),
                event.event_version,
                datetime.now(),
            )
            return event.event_id

    async def append_events(
        self,
        novel_id: str,
        events: List[NarrativeEvent],
        volume_num: int = None,
        chapter_num: int = None,
        scene_id: int = None,
    ) -> List[str]:
        """批量存储事件（事务内）"""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                uuids = []
                for event in events:
                    uuid = await self.append_event(
                        novel_id, event, volume_num, chapter_num, scene_id
                    )
                    uuids.append(uuid)
                return uuids

    async def get_events_since(
        self,
        novel_id: str,
        since_event_id: int = 0,
        limit: int = 1000,
    ) -> List[Tuple[int, NarrativeEvent]]:
        """返回 (event_db_id, event) 列表"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, event_uuid, event_type, event_data, event_version, timestamp
                FROM narrative_events
                WHERE novel_id = $1 AND id > $2
                ORDER BY id ASC
                LIMIT $3
                """,
                novel_id,
                since_event_id,
                limit,
            )
            result = []
            for row in rows:
                event_data = row["event_data"]
                if isinstance(event_data, str):
                    event_data = json.loads(event_data)
                event_data = EventUpcaster.upcast(event_data)
                event = event_from_dict(row["event_type"], event_data)
                if event:
                    result.append((row["id"], event))
            return result

    async def get_chapter_events(
        self,
        novel_id: str,
        volume_num: int,
        chapter_num: int,
    ) -> List[NarrativeEvent]:
        """获取某一章的所有事件（只返回事件对象）"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, event_type, event_data, event_version
                FROM narrative_events
                WHERE novel_id = $1 AND volume_num = $2 AND chapter_num = $3
                ORDER BY id ASC
                """,
                novel_id,
                volume_num,
                chapter_num,
            )
            events = []
            for row in rows:
                event_data = row["event_data"]
                if isinstance(event_data, str):
                    event_data = json.loads(event_data)
                event_data = EventUpcaster.upcast(event_data)
                event = event_from_dict(row["event_type"], event_data)
                if event:
                    events.append(event)
            return events

    async def get_last_event_id(self, novel_id: str) -> int:
        """获取最新事件的数据库 ID"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT MAX(id) as last_id FROM narrative_events WHERE novel_id = $1",
                novel_id,
            )
            return row["last_id"] or 0

    async def truncate_events_after(self, novel_id: str, event_db_id: int):
        """删除该事件之后的所有事件（不包含当前事件）"""
        async with self.pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM narrative_events WHERE novel_id = $1 AND id > $2",
                novel_id,
                event_db_id,
            )