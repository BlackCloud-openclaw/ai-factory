#!/usr/bin/env python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import asyncpg
from src.config import config

async def clean():
    dsn = config.postgres_dsn
    conn = await asyncpg.connect(dsn)
    try:
        result1 = await conn.execute("DELETE FROM events WHERE novel_id LIKE 'test_novel%'")
        print(f"Deleted {result1.split()[-1]} rows from events (test_novel)")
        result2 = await conn.execute("DELETE FROM events WHERE novel_id LIKE 'e2e%'")
        print(f"Deleted {result2.split()[-1]} rows from events (e2e)")
        result3 = await conn.execute("DELETE FROM novels WHERE novel_id LIKE 'test_snap%'")
        print(f"Deleted {result3.split()[-1]} rows from novels (test_snap)")
        result4 = await conn.execute("DELETE FROM novels WHERE novel_id LIKE 'e2e%'")
        print(f"Deleted {result4.split()[-1]} rows from novels (e2e)")
        print("✅ Cleaned all test data.")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(clean())