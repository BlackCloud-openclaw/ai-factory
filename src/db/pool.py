# src/db/pool.py
import asyncpg
from typing import Optional, Dict, Any
from src.config import config
from src.common.logging import setup_logging

logger = setup_logging("db.pool")

_db_pool = None


def get_db_pool():
    """返回全局数据库连接池（供其他模块使用）"""
    return _db_pool


async def init_db_pool():
    """初始化数据库连接池（在应用启动时调用）"""
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
    """关闭数据库连接池（在应用关闭时调用）"""
    global _db_pool
    if _db_pool:
        await _db_pool.close()
        _db_pool = None
        logger.info("Database pool closed")


# ==================== 写作进度管理（writing_progress 表） ====================

async def init_writing_progress(
    novel_id: str,
    volume: int = 1,
    chapter: int = 1,
    scene: int = 0,
    chapter_completed: bool = False
) -> None:
    """
    初始化或重置小说的写作进度记录。
    如果记录已存在，则更新为给定值（幂等）。
    """
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
        logger.info(f"init_writing_progress: novel={novel_id}, volume={volume}, chapter={chapter}, scene={scene}, completed={chapter_completed}")


async def update_progress_scene(
    novel_id: str,
    scene_index: int,
    chapter_completed: Optional[bool] = None
) -> None:
    """
    更新当前场景索引，可选同时更新章节完成标志。
    
    Args:
        novel_id: 小说ID
        scene_index: 新的场景索引（已完成场景数，0‑based）
        chapter_completed: 如果提供，则同时更新本章是否完成标志；若为 None 则不修改该字段
    """
    pool = get_db_pool()
    async with pool.acquire() as conn:
        if chapter_completed is None:
            await conn.execute("""
                UPDATE writing_progress
                SET current_scene = $2, last_updated = NOW()
                WHERE project_id = $1
            """, novel_id, scene_index)
            logger.info(f"update_progress_scene: novel={novel_id}, scene={scene_index} (no chapter_completed update)")
        else:
            await conn.execute("""
                UPDATE writing_progress
                SET current_scene = $2, chapter_completed = $3, last_updated = NOW()
                WHERE project_id = $1
            """, novel_id, scene_index, chapter_completed)
            logger.info(f"update_progress_scene: novel={novel_id}, scene={scene_index}, chapter_completed={chapter_completed}")


async def update_progress_chapter(
    novel_id: str,
    new_chapter: int,
    reset_scene: bool = True
) -> None:
    """
    更新当前章节号，并重置场景索引为0，清除章节完成标志。
    
    Args:
        novel_id: 小说ID
        new_chapter: 新的章节号
        reset_scene: 是否重置场景索引（默认 True）
    """
    pool = get_db_pool()
    async with pool.acquire() as conn:
        if reset_scene:
            await conn.execute("""
                UPDATE writing_progress
                SET current_chapter = $2, current_scene = 0, chapter_completed = FALSE, last_updated = NOW()
                WHERE project_id = $1
            """, novel_id, new_chapter)
            logger.info(f"update_progress_chapter: novel={novel_id}, chapter={new_chapter}, scene reset to 0")
        else:
            await conn.execute("""
                UPDATE writing_progress
                SET current_chapter = $2, chapter_completed = FALSE, last_updated = NOW()
                WHERE project_id = $1
            """, novel_id, new_chapter)
            logger.info(f"update_progress_chapter: novel={novel_id}, chapter={new_chapter} (scene unchanged)")


async def update_progress_volume(
    novel_id: str,
    new_volume: int,
) -> None:
    """
    更新当前卷号，并重置章节为1，场景索引为0，清除章节完成标志。
    
    Args:
        novel_id: 小说ID
        new_volume: 新的卷号
    """
    pool = get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            UPDATE writing_progress
            SET current_volume = $2, current_chapter = 1, current_scene = 0, chapter_completed = FALSE, last_updated = NOW()
            WHERE project_id = $1
        """, novel_id, new_volume)
        logger.info(f"update_progress_volume: novel={novel_id}, volume={new_volume}, chapter reset to 1, scene to 0")


async def load_writing_progress(novel_id: str) -> Optional[Dict[str, Any]]:
    """
    加载小说的写作进度。
    
    Returns:
        字典，包含 current_volume, current_chapter, current_scene, chapter_completed, last_updated
        如果记录不存在则返回 None
    """
    pool = get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT current_volume, current_chapter, current_scene, chapter_completed, last_updated "
            "FROM writing_progress WHERE project_id = $1",
            novel_id
        )
    return dict(row) if row else None