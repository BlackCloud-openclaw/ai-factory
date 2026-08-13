"""一致性预算管理 - 数据库持久化（事务锁+同步实例属性）"""
import logging
from src.db import get_db_pool

logger = logging.getLogger(__name__)


class ConsistencyBudget:
    def __init__(self, novel_id: str, volume_num: int, chapter_num: int):
        self.novel_id = novel_id
        self.volume_num = volume_num
        self.chapter_num = chapter_num
        self.max_warnings = 3
        self.max_soft = 1
        self.remaining_warnings = self.max_warnings
        self.remaining_soft = self.max_soft

    async def load(self) -> None:
        # 如果 novel_id 为空，直接返回
        if not self.novel_id:
            return
        pool = get_db_pool()
        if pool is None:
            return
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT remaining_warnings, remaining_soft
                FROM chapter_budget
                WHERE novel_id = $1 AND volume_num = $2 AND chapter_num = $3
                """,
                self.novel_id, self.volume_num, self.chapter_num
            )
            if row:
                self.remaining_warnings = row["remaining_warnings"]
                self.remaining_soft = row["remaining_soft"]
            else:
                self.remaining_warnings = self.max_warnings
                self.remaining_soft = self.max_soft

    async def reset(self) -> None:
        pool = get_db_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO chapter_budget (novel_id, volume_num, chapter_num, remaining_warnings, remaining_soft)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (novel_id, volume_num, chapter_num) DO UPDATE
                    SET remaining_warnings = EXCLUDED.remaining_warnings,
                        remaining_soft = EXCLUDED.remaining_soft
                    """,
                    self.novel_id, self.volume_num, self.chapter_num,
                    self.max_warnings, self.max_soft
                )
        self.remaining_warnings = self.max_warnings
        self.remaining_soft = self.max_soft

    async def consume(self, severity: str) -> bool:
        pool = get_db_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    SELECT remaining_warnings, remaining_soft
                    FROM chapter_budget
                    WHERE novel_id = $1 AND volume_num = $2 AND chapter_num = $3
                    FOR UPDATE
                    """,
                    self.novel_id, self.volume_num, self.chapter_num
                )
                if not row:
                    remaining_warnings = self.max_warnings
                    remaining_soft = self.max_soft
                    await conn.execute(
                        """
                        INSERT INTO chapter_budget (novel_id, volume_num, chapter_num, remaining_warnings, remaining_soft)
                        VALUES ($1, $2, $3, $4, $5)
                        """,
                        self.novel_id, self.volume_num, self.chapter_num,
                        remaining_warnings, remaining_soft
                    )
                else:
                    remaining_warnings = row["remaining_warnings"]
                    remaining_soft = row["remaining_soft"]

                if severity == "warning":
                    if remaining_warnings > 0:
                        remaining_warnings -= 1
                        await conn.execute(
                            """
                            UPDATE chapter_budget
                            SET remaining_warnings = $1
                            WHERE novel_id = $2 AND volume_num = $3 AND chapter_num = $4
                            """,
                            remaining_warnings, self.novel_id, self.volume_num, self.chapter_num
                        )
                        self.remaining_warnings = remaining_warnings
                        return True
                    else:
                        return False
                elif severity == "soft_contradiction":
                    if remaining_soft > 0:
                        remaining_soft -= 1
                        await conn.execute(
                            """
                            UPDATE chapter_budget
                            SET remaining_soft = $1
                            WHERE novel_id = $2 AND volume_num = $3 AND chapter_num = $4
                            """,
                            remaining_soft, self.novel_id, self.volume_num, self.chapter_num
                        )
                        self.remaining_soft = remaining_soft
                        return True
                    else:
                        return False
                return True