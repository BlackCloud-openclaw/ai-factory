# src/db/pool.py
import asyncpg
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