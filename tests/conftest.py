import pytest
import asyncio
from src.db import init_db_pool, close_db_pool, get_db_pool
from src.config import config
import asyncpg

TEST_DB_NAME = "ai_factory_test"

# 使用同步 fixture 包装异步初始化（避免 pytest-asyncio 收集问题）
@pytest.fixture(scope="session")
def db_pool():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    async def _setup():
        sys_conn = await asyncpg.connect(
            host=config.postgres_host,
            port=config.postgres_port,
            user=config.postgres_user,
            password=config.postgres_password,
            database="postgres"
        )
        try:
            await sys_conn.execute(f"CREATE DATABASE {TEST_DB_NAME} OWNER {config.postgres_user}")
        except asyncpg.exceptions.DuplicateDatabaseError:
            pass
        finally:
            await sys_conn.close()
        
        original_db = config.postgres_db
        config.postgres_db = TEST_DB_NAME
        await init_db_pool()
        pool = get_db_pool()
        
        from pathlib import Path
        sql_path = Path("scripts/init_novel_db.sql")
        if sql_path.exists():
            sql = sql_path.read_text()
            async with pool.acquire() as conn:
                await conn.execute(sql)
        else:
            raise FileNotFoundError(f"Schema not found: {sql_path}")
        
        return pool, original_db
    
    pool, original_db = loop.run_until_complete(_setup())
    yield pool
    
    async def _teardown():
        await close_db_pool()
        config.postgres_db = original_db
        sys_conn = await asyncpg.connect(
            host=config.postgres_host,
            port=config.postgres_port,
            user=config.postgres_user,
            password=config.postgres_password,
            database="postgres"
        )
        await sys_conn.execute(f"DROP DATABASE {TEST_DB_NAME} WITH (FORCE)")
        await sys_conn.close()
    
    loop.run_until_complete(_teardown())
    loop.close()

@pytest.fixture
def novel_id():
    return "test_e2e_novel"