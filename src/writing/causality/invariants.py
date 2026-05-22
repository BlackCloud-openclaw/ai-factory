"""投影不变量检查器"""
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# 单值关系列表
SINGLETON_RELATIONS = {"realm", "is_alive", "location"}


class InvariantChecker:
    def __init__(self, pool):
        self.pool = pool

    async def check_after_apply(self, novel_id: str, affected_subjects: List[str]):
        """应用 Predicate Delta 后检查不变量"""
        for subject in affected_subjects:
            await self._check_singleton_relations(novel_id, subject)
            await self._check_confidence_range(novel_id, subject)
        await self._check_temporal_order(novel_id)

    async def _check_singleton_relations(self, novel_id: str, subject: str):
        """检查单值关系是否只有一条活跃记录"""
        async with self.pool.acquire() as conn:
            for relation in SINGLETON_RELATIONS:
                count = await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM predicates
                    WHERE novel_id = $1 AND subject = $2 AND relation = $3 AND is_active = true
                    """,
                    novel_id, subject, relation
                )
                if count > 1:
                    logger.error(
                        f"Invariant violation: {subject}.{relation} has {count} active predicates"
                    )
                    # 可选：自动修复或触发告警

    async def _check_confidence_range(self, novel_id: str, subject: str):
        """检查置信度范围"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT confidence FROM predicates
                WHERE novel_id = $1 AND subject = $2 AND is_active = true
                """,
                novel_id, subject
            )
            for row in rows:
                conf = row["confidence"]
                if not (0.0 <= conf <= 1.0):
                    logger.error(f"Invariant violation: confidence {conf} out of range")

    async def _check_temporal_order(self, novel_id: str):
        """检查 valid_from_event_id <= valid_to_event_id"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, valid_from_event_id, valid_to_event_id FROM predicates
                WHERE novel_id = $1 AND valid_to_event_id IS NOT NULL
                """,
                novel_id
            )
            for row in rows:
                if row["valid_from_event_id"] > row["valid_to_event_id"]:
                    logger.error(
                        f"Invariant violation: predicate {row['id']} has from > to"
                    )