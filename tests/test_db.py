# 测试脚本（单独运行）
import asyncio
from src.db.pool import init_db_pool, save_progress, load_progress

async def test():
    await init_db_pool()
    await save_progress("test_project", 1, 2, 3, True)
    progress = await load_progress("test_project")
    print(progress)

asyncio.run(test())