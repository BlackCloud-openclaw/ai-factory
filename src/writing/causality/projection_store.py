"""ProjectionStore - 将 PredicateDelta 应用到数据库（幂等、冲突处理、死信）"""
import json
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from src.db import get_db_pool
from .predicate import Predicate
from .delta import PredicateDelta, PredicateRef

logger = logging.getLogger(__name__)


class ProjectionStore:
    # 单值关系列表（与 DeltaEngine 保持一致）
    SINGLETON_RELATIONS = {"realm", "is_alive", "location"}
    
    def __init__(self, pool=None):
        self.pool = pool or get_db_pool()
        if not self.pool:
            raise RuntimeError("Database pool not initialized")

    async def apply_delta(self, delta: PredicateDelta) -> bool:
        """
        应用 PredicateDelta 到 predicates 表。
        返回 True 表示成功，False 表示应进入死信。
        """
        delta_id = self._make_delta_id(delta.novel_id, delta.event_id, delta.projection_version)

        # 1. 幂等检查
        if await self._is_applied(delta_id):
            logger.debug(f"Delta {delta_id} already applied, skipping")
            return True

        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    # 2. 失效谓词
                    for ref in delta.to_deactivate:
                        await self._deactivate_predicate(conn, delta.novel_id, ref)

                    # 3. 激活谓词（处理单值冲突）
                    for pred in delta.to_activate:
                        await self._activate_predicate(conn, delta.novel_id, pred, delta.event_id)

                    # 4. 记录已应用
                    await conn.execute(
                        "INSERT INTO projection_applied (delta_id, novel_id, event_id) VALUES ($1, $2, $3)",
                        delta_id, delta.novel_id, delta.event_id
                    )
            return True
        except Exception as e:
            logger.error(f"Failed to apply delta {delta_id}: {e}", exc_info=True)
            # 写入死信队列（不阻塞）
            await self._record_dead_letter(delta.novel_id, delta.event_id, str(e))
            return False

    async def _is_applied(self, delta_id: str) -> bool:
        async with self.pool.acquire() as conn:
            row = await conn.fetchval(
                "SELECT 1 FROM projection_applied WHERE delta_id = $1",
                delta_id
            )
            return row is not None

    async def _deactivate_predicate(self, conn, novel_id: str, ref: PredicateRef):
        # 将指定 identity 的活跃谓词标记为失效
        await conn.execute(
            """
            UPDATE predicates
            SET valid_to_event_id = $1, is_active = false, updated_at = NOW()
            WHERE novel_id = $2 AND identity_key = $3 AND is_active = true
            """,
            ref.event_id, novel_id, ref.identity_key
        )

    async def _activate_predicate(self, conn, novel_id: str, pred: Predicate, event_id: int):
        # 单值关系：先失效同 subject、relation 的旧记录
        if pred.relation in self.SINGLETON_RELATIONS:
            old_rows = await conn.fetch(
                """
                SELECT id, subject, relation, object
                FROM predicates
                WHERE novel_id = $1 AND subject = $2 AND relation = $3 AND is_active = true
                """,
                novel_id, pred.subject, pred.relation
            )
            for row in old_rows:
                logger.info(f"Deactivating old {pred.relation} predicate: subject={row['subject']}, "
                            f"object={row['object']} (event_id={event_id})")
                await conn.execute(
                    """
                    UPDATE predicates
                    SET valid_to_event_id = $1, is_active = false, updated_at = NOW()
                    WHERE id = $2
                    """,
                    event_id, row["id"]
                )

        # 插入新谓词
        object_json = json.dumps(pred.object, ensure_ascii=False)
        identity_key = pred.identity_key()  # 使用 Predicate 已有的方法
        logger.debug(f"Activating predicate: subject={pred.subject}, relation={pred.relation}, "
                    f"object={pred.object}, identity_key={identity_key}")
        await conn.execute(
            """
            INSERT INTO predicates
            (novel_id, event_id, source_event_type, source_event_semantic, projection_version,
            subject, relation, object, negated, confidence, priority, scope,
            is_active, valid_from_event_id, valid_to_event_id, identity_key)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, true, $2, NULL, $13)
            """,
            novel_id,
            event_id,
            pred.source_event_type or '',
            pred.source_event_semantic or '',
            1,
            pred.subject,
            pred.relation,
            object_json,
            pred.negated,
            pred.confidence,
            pred.priority,
            pred.scope,
            identity_key
        )

    async def _record_dead_letter(self, novel_id: str, event_id: int, error: str):
        async with self.pool.acquire() as conn:
            # 检查该事件是否已存在死信记录
            existing = await conn.fetchval(
                "SELECT retry_count FROM projection_dead_letters WHERE novel_id=$1 AND event_id=$2",
                novel_id, event_id
            )
            retry = (existing or 0) + 1
            await conn.execute("""
                INSERT INTO projection_dead_letters (novel_id, event_id, error, retry_count, status)
                VALUES ($1, $2, $3, $4, 'pending')
                ON CONFLICT (novel_id, event_id) DO UPDATE
                SET retry_count = EXCLUDED.retry_count, error = EXCLUDED.error, status = 'pending', created_at = NOW()
            """, novel_id, event_id, error, retry)
            
            # 告警：首次发生或重试超过阈值
            if retry == 1:
                logger.error(f"[DEAD_LETTER] New dead letter for novel {novel_id}, event {event_id}: {error[:200]}")
            elif retry >= 3:
                logger.critical(f"[DEAD_LETTER] Persistent dead letter (retry {retry}) for novel {novel_id}, event {event_id}: {error[:200]}")

    @staticmethod
    def _make_delta_id(novel_id: str, event_id: int, version: int) -> str:
        return f"{novel_id}_init_v{version}" if event_id == 0 else f"{novel_id}_{event_id}_v{version}"
