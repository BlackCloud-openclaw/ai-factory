# src/db/pool.py (修改后完整内容)
import asyncpg
from typing import Optional, Dict, Any
from src.config import config
from src.common.logging import setup_logging

logger = setup_logging("db.pool")

_db_pool = None


def get_db_pool():
    return _db_pool


async def init_db_pool():
    global _db_pool
    if _db_pool is None:
        _db_pool = await asyncpg.create_pool(
            config.postgres_dsn,
            min_size=1,
            max_size=10,
            command_timeout=60
        )
        logger.info("Database pool created")
    return _db_pool


async def close_db_pool():
    global _db_pool
    if _db_pool:
        await _db_pool.close()
        _db_pool = None
        logger.info("Database pool closed")


# ==================== 写作进度管理（带并发保护） ====================

async def init_writing_progress(
    novel_id: str,
    volume: int = 1,
    chapter: int = 1,
    scene: int = 0,
    chapter_completed: bool = False
) -> None:
    pool = get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO writing_progress (project_id, current_volume, current_chapter, current_scene, chapter_completed, last_updated)
            VALUES ($1, $2, $3, $4, $5, NOW())
            ON CONFLICT (project_id) DO UPDATE SET
                current_volume = EXCLUDED.current_volume,
                current_chapter = EXCLUDED.current_chapter,
                current_scene = EXCLUDED.current_scene,
                chapter_completed = EXCLUDED.chapter_completed,
                last_updated = NOW()
        """, novel_id, volume, chapter, scene, chapter_completed)
        logger.info(f"init_writing_progress: {novel_id} v{volume}c{chapter}s{scene}")


async def update_progress_scene(
    novel_id: str,
    scene_index: int,
    chapter_completed: Optional[bool] = None
) -> None:
    """
    更新当前场景索引，使用行锁保证单调递增。
    """
    pool = get_db_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            # 锁定行
            row = await conn.fetchrow(
                "SELECT current_volume, current_chapter, current_scene, chapter_completed "
                "FROM writing_progress WHERE project_id = $1 FOR UPDATE",
                novel_id
            )
            if not row:
                # 若不存在，先初始化（幂等）
                await init_writing_progress(novel_id)
                row = await conn.fetchrow(
                    "SELECT current_volume, current_chapter, current_scene, chapter_completed "
                    "FROM writing_progress WHERE project_id = $1 FOR UPDATE",
                    novel_id
                )
            old_scene = row["current_scene"]
            new_scene = max(old_scene, scene_index)  # 单调递增

            if chapter_completed is None:
                await conn.execute(
                    "UPDATE writing_progress SET current_scene = $1, last_updated = NOW() WHERE project_id = $2",
                    new_scene, novel_id
                )
                logger.info(f"update_progress_scene: {novel_id} scene {old_scene} -> {new_scene}")
            else:
                # 章节完成标志只能从 false 变为 true，不能逆转
                old_completed = row["chapter_completed"]
                new_completed = chapter_completed or old_completed
                await conn.execute(
                    "UPDATE writing_progress SET current_scene = $1, chapter_completed = $2, last_updated = NOW() "
                    "WHERE project_id = $3",
                    new_scene, new_completed, novel_id
                )
                logger.info(f"update_progress_scene: {novel_id} scene {old_scene} -> {new_scene}, completed {old_completed} -> {new_completed}")


async def update_progress_chapter(
    novel_id: str,
    new_chapter: int,
    reset_scene: bool = True,
    require_sync: bool = True
) -> None:
    """
    更新当前章节，使用行锁保证单调递增。
    """
    pool = get_db_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                "SELECT current_chapter, current_scene FROM writing_progress WHERE project_id = $1 FOR UPDATE",
                novel_id
            )
            if not row:
                await init_writing_progress(novel_id)
                row = await conn.fetchrow("SELECT current_chapter, current_scene FROM writing_progress WHERE project_id = $1 FOR UPDATE", novel_id)
            old_chapter = row["current_chapter"]
            new_chapter_val = max(old_chapter, new_chapter)  # 单调递增
            new_scene = 0 if reset_scene else row["current_scene"]

            await conn.execute(
                "UPDATE writing_progress SET current_chapter = $1, current_scene = $2, chapter_completed = FALSE, last_updated = NOW() "
                "WHERE project_id = $3",
                new_chapter_val, new_scene, novel_id
            )
            logger.info(f"update_progress_chapter: {novel_id} chapter {old_chapter} -> {new_chapter_val}, scene reset to {new_scene}")


async def update_progress_volume(
    novel_id: str,
    new_volume: int,
) -> None:
    """
    更新当前卷，使用行锁保证单调递增，并重置章节和场景。
    """
    pool = get_db_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                "SELECT current_volume FROM writing_progress WHERE project_id = $1 FOR UPDATE",
                novel_id
            )
            if not row:
                await init_writing_progress(novel_id)
                row = await conn.fetchrow("SELECT current_volume FROM writing_progress WHERE project_id = $1 FOR UPDATE", novel_id)
            old_volume = row["current_volume"]
            new_volume_val = max(old_volume, new_volume)

            await conn.execute(
                "UPDATE writing_progress SET current_volume = $1, current_chapter = 1, current_scene = 0, chapter_completed = FALSE, last_updated = NOW() "
                "WHERE project_id = $2",
                new_volume_val, novel_id
            )
            logger.info(f"update_progress_volume: {novel_id} volume {old_volume} -> {new_volume_val}, reset chapter to 1, scene to 0")


async def load_writing_progress(novel_id: str) -> Optional[Dict[str, Any]]:
    pool = get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT current_volume, current_chapter, current_scene, chapter_completed, last_updated "
            "FROM writing_progress WHERE project_id = $1",
            novel_id
        )
    return dict(row) if row else None