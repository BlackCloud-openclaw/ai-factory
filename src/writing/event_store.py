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
import asyncio
from datetime import datetime
from typing import Dict, Optional, List, Tuple

from .events import NarrativeEvent, event_to_dict, event_from_dict
from src.common.logging import setup_logging
from src.writing.causality.projector import DeltaEngine
from src.writing.causality.scheduler import ProjectionScheduler
from src.writing.causality.predicate import Predicate
from src.writing.causality.upcaster import LATEST_EVENT_SCHEMA_VERSION, upcast_event_envelope
from src.writing.narrative_projection import NarrativeProjector
from src.db import get_db_pool

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
        conn: Optional[asyncpg.Connection] = None,  # 新增
    ) -> str:
        """存储单个事件（幂等），返回事件UUID"""
        if conn is None:
            async with self.pool.acquire() as conn:
                return await self._append_event(
                    conn, novel_id, event, volume_num, chapter_num, scene_id
                )
        else:
            return await self._append_event(
                conn, novel_id, event, volume_num, chapter_num, scene_id
            )

    async def _append_event(
        self,
        conn: asyncpg.Connection,
        novel_id: str,
        event: NarrativeEvent,
        volume_num: int = None,
        chapter_num: int = None,
        scene_id: int = None,
    ) -> str:
        """内部实现，使用传入的连接"""
        logger.info(f"[DEBUG] _append_event called for novel {novel_id}, event type {event.type}")
        event_data = event_to_dict(event)
        event_data["event_schema_version"] = LATEST_EVENT_SCHEMA_VERSION
        event_data = upcast_event_envelope(event_data)

        row = await conn.fetchrow(
            """
            INSERT INTO narrative_events 
            (event_uuid, novel_id, volume_num, chapter_num, scene_id, 
            event_type, event_data, event_version, timestamp)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (event_uuid) DO UPDATE SET event_uuid = EXCLUDED.event_uuid
            RETURNING id
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
        event_db_id = row["id"] if row else None

        # 触发投影（同步，可后续改为异步队列）
        try:
            delta_engine = DeltaEngine()
            scheduler = ProjectionScheduler()
            current_active = await self._load_active_predicates(novel_id, conn)

            # 构建事件字典：复制 upcast 后的数据，补充必要字段
            event_dict = event_data.copy()
            event_dict["novel_id"] = novel_id
            event_dict["id"] = event_db_id
            event_dict["event_id"] = event_db_id
            if "semantic" not in event_dict:
                event_dict["semantic"] = "state_mutation"

            delta = delta_engine.compute_delta(current_active, event_dict)
            # 投影调度器内部会使用自己的连接，这里不传入 conn（投影可以独立提交）
            await scheduler.schedule(novel_id, delta)
        except Exception as e:
            logger.error(f"Projection failed for event {event.event_id}: {e}", exc_info=True)

        import sys
        sys.stderr.write(f"type(NarrativeProjector) = {type(NarrativeProjector)}\n")
        sys.stderr.flush()       

        return event.event_id

    async def append_events(
        self,
        novel_id: str,
        events: List[NarrativeEvent],
        volume_num: int = None,
        chapter_num: int = None,
        scene_id: int = None,
        conn: Optional[asyncpg.Connection] = None,
    ) -> List[str]:
        """批量存储事件（事务内）"""
        if conn is None:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    return await self._append_events_batch(
                        conn, novel_id, events, volume_num, chapter_num, scene_id
                    )
        else:
            return await self._append_events_batch(
                conn, novel_id, events, volume_num, chapter_num, scene_id
            )

    async def _append_events_batch(
        self,
        conn: asyncpg.Connection,
        novel_id: str,
        events: List[NarrativeEvent],
        volume_num: int = None,
        chapter_num: int = None,
        scene_id: int = None,
    ) -> List[str]:
        uuids = []
        for event in events:
            uuid = await self._append_event(
                conn, novel_id, event, volume_num, chapter_num, scene_id
            )
            uuids.append(uuid)
        return uuids

    async def get_events_since(
        self,
        novel_id: str,
        since_event_id: int = 0,
        limit: int = 1000,
    ) -> List[Tuple[int, NarrativeEvent]]:
        """返回 (event_db_id, event) 列表，事件已升级到最新 schema"""
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
                event_data = upcast_event_envelope(event_data)
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
                event_data = upcast_event_envelope(event_data)
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

    async def _load_active_predicates(
        self, novel_id: str, conn: Optional[asyncpg.Connection] = None
    ) -> Dict[str, Predicate]:
        """加载当前小说的所有活跃谓词（用于 delta 计算）"""
        if conn is None:
            async with self.pool.acquire() as conn:
                return await self._load_active_predicates(novel_id, conn)
        rows = await conn.fetch(
            """
            SELECT subject, relation, object, negated, confidence, priority, scope,
                event_id AS source_event_id, source_event_type, source_event_semantic
            FROM predicates
            WHERE novel_id = $1 AND is_active = true
            """,
            novel_id
        )
        predicates = {}
        for row in rows:
            obj = row["object"]
            if isinstance(obj, str):
                try:
                    obj = json.loads(obj)
                except:
                    pass

            pred = Predicate(
                subject=row["subject"],
                relation=row["relation"],
                object=obj,
                negated=row["negated"],
                confidence=row["confidence"],
                priority=row["priority"],
                scope=row["scope"],
                source_event_id=row["source_event_id"],
                source_event_type=row["source_event_type"],
                source_event_semantic=row["source_event_semantic"]
            )
            predicates[pred.identity_key()] = pred
        return predicates