import pytest
from src.db import init_db_pool, close_db_pool, get_db_pool

@pytest.fixture(scope="session")
async def db_pool():
    await init_db_pool()
    pool = get_db_pool()
    yield pool
    await close_db_pool()

@pytest.fixture
def novel_id():
    return "test_e2e_novel"