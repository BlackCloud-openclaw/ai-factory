#!/usr/bin/env python
"""Initialize novel-related tables using asyncpg."""
import asyncio
import asyncpg
from pathlib import Path

async def init_tables():
    # 数据库连接参数（与 config 保持一致）
    dsn = "postgresql://woami:kali@localhost:5432/ai_factory"
    
    # 读取 SQL 文件
    sql_file = Path(__file__).parent / "init_novel_db.sql"
    if not sql_file.exists():
        print(f"SQL file not found: {sql_file}")
        return
    
    sql = sql_file.read_text()
    
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(sql)
        print("✅ Novel tables created/verified successfully.")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(init_tables())