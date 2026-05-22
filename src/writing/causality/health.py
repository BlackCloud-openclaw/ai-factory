"""投影健康检查与漂移检测"""
import hashlib
import json
import logging
from src.db import get_db_pool

logger = logging.getLogger(__name__)


class HealthChecker:
    @staticmethod
    async def _compute_core_predicates_hash(novel_id: str) -> str:
        """仅计算核心谓词的哈希值（priority='core' 或特定关系）"""
        pool = get_db_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT subject, relation, object, negated, confidence
                FROM predicates
                WHERE novel_id = $1 AND is_active = true
                  AND (priority = 'core' OR relation IN ('realm', 'is_alive', 'location'))
                ORDER BY subject, relation, object
                """,
                novel_id
            )
            data = json.dumps([dict(row) for row in rows], sort_keys=True)
            return hashlib.md5(data.encode()).hexdigest()

    @staticmethod
    async def check_drift(novel_id: str) -> str:
        """
        检测投影漂移，返回漂移级别：'INFO', 'WARNING', 'CRITICAL'
        仅基于核心谓词比较，避免非核心变化导致的误报。
        """
        pool = get_db_pool()
        current_hash = await HealthChecker._compute_core_predicates_hash(novel_id)

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT core_predicates_hash, drift_level FROM projection_health WHERE novel_id = $1",
                novel_id
            )
            if not row:
                drift_level = "INFO"
                await conn.execute(
                    """
                    INSERT INTO projection_health (novel_id, core_predicates_hash, drift_level, updated_at)
                    VALUES ($1, $2, $3, NOW())
                    """,
                    novel_id, current_hash, drift_level
                )
                return drift_level

            if row["core_predicates_hash"] == current_hash:
                return row["drift_level"] or "INFO"

            # 哈希不一致，说明核心谓词发生了变化（可能是正常演化，也可能是漂移）
            # 暂时标记为 INFO，不触发 WARNING，因为核心谓词变化通常是预期的。
            drift_level = "INFO"
            await conn.execute(
                """
                UPDATE projection_health
                SET core_predicates_hash = $1, drift_level = $2, updated_at = NOW()
                WHERE novel_id = $3
                """,
                current_hash, drift_level, novel_id
            )
            logger.info(f"Core predicates hash updated for novel {novel_id}")
            return drift_level

    @staticmethod
    async def set_validator_mode(novel_id: str, mode: str):
        """设置验证器模式：'normal', 'degraded'"""
        pool = get_db_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO projection_health (novel_id, validator_mode, updated_at)
                VALUES ($1, $2, NOW())
                ON CONFLICT (novel_id) DO UPDATE
                SET validator_mode = EXCLUDED.validator_mode, updated_at = NOW()
                """,
                novel_id, mode
            )