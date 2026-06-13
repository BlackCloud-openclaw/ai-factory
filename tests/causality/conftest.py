import pytest
import pytest_asyncio
import asyncpg
from src.db import init_db_pool, close_db_pool, get_db_pool
from src.config import config
from pathlib import Path

TEST_DB_NAME = "ai_factory_test"

@pytest_asyncio.fixture(scope="session")
async def db_pool():
    """创建独立测试数据库，session 级别"""
    # 连接 postgres 系统数据库
    sys_conn = await asyncpg.connect(
        host=config.postgres_host,
        port=config.postgres_port,
        user=config.postgres_user,
        password=config.postgres_password,
        database="postgres"
    )
    try:
        await sys_conn.execute(f"CREATE DATABASE {TEST_DB_NAME} OWNER {config.postgres_user}")
        print(f"✅ Created test database {TEST_DB_NAME}")
    except asyncpg.exceptions.DuplicateDatabaseError:
        print(f"ℹ️ Test database {TEST_DB_NAME} already exists")
    finally:
        await sys_conn.close()

    # 修改配置并初始化连接池
    original_db = config.postgres_db
    config.postgres_db = TEST_DB_NAME
    await init_db_pool()
    pool = get_db_pool()

    # 初始化 schema（复用 init_novel_db.sql）
    sql_file = Path("scripts/init_novel_db.sql")
    if sql_file.exists():
        sql = sql_file.read_text()
        async with pool.acquire() as conn:
            await conn.execute(sql)
        print("✅ Test schema initialized")
    else:
        raise FileNotFoundError(f"Schema file not found: {sql_file}")

    yield pool

    # 清理
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
    print(f"🗑️ Dropped test database {TEST_DB_NAME}")

@pytest_asyncio.fixture
async def tx_conn(db_pool):
    """每个测试函数使用独立事务，自动回滚"""
    async with db_pool.acquire() as conn:
        async with conn.transaction():
            yield conn

@pytest.fixture
def novel_id():
    return "test_novel_001"