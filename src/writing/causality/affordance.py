"""叙事可供性管理 - 冷却与新颖度"""
from src.db import get_db_pool


async def record_affordance_usage(novel_id: str, affordance_id: str, chapter_num: int):
    pool = get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO affordance_usage (novel_id, affordance_id, last_used_chapter)
            VALUES ($1, $2, $3)
            ON CONFLICT (novel_id, affordance_id) DO UPDATE
            SET last_used_chapter = EXCLUDED.last_used_chapter
            """,
            novel_id, affordance_id, chapter_num
        )


async def get_affordance_cooldown_penalty(novel_id: str, affordance_id: str, current_chapter: int, cooldown: int) -> float:
    if cooldown <= 0:
        return 1.0
    pool = get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT last_used_chapter FROM affordance_usage WHERE novel_id=$1 AND affordance_id=$2",
            novel_id, affordance_id
        )
        if row and (current_chapter - row["last_used_chapter"]) < cooldown:
            return 0.2
    return 1.0