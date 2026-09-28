import sys
import os
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
from dotenv import load_dotenv
load_dotenv(project_root / ".env")
import asyncio
from src.db import init_db_pool, get_db_pool

async def main():
    await init_db_pool()
    pool = get_db_pool()
    if not pool:
        print("No pool")
        return
    async with pool.acquire() as conn:
        total = await conn.fetchval("SELECT COUNT(*) FROM shadow_rewrite_log")
        print(f"Total records: {total}")
        
        quadrants = await conn.fetch("""
            SELECT original_passed, rewritten_passed, COUNT(*) 
            FROM shadow_rewrite_log 
            GROUP BY original_passed, rewritten_passed
        """)
        print("Quadrant distribution:")
        for row in quadrants:
            print(f"  original_passed={row['original_passed']}, rewritten_passed={row['rewritten_passed']}: {row['count']}")
        
        d_status = await conn.fetch("""
            SELECT status, COUNT(*) 
            FROM shadow_rewrite_log 
            WHERE original_passed = false AND rewritten_passed = false
            GROUP BY status
        """)
        print("D quadrant status distribution:")
        for row in d_status:
            print(f"  status={row['status']}: {row['count']}")
        
        recent_d = await conn.fetch("""
            SELECT id, scene_id, status, executed_at
            FROM shadow_rewrite_log 
            WHERE original_passed = false AND rewritten_passed = false
            ORDER BY executed_at DESC
            LIMIT 5
        """)
        print("Recent 5 D quadrant records:")
        for r in recent_d:
            print(f"  id={r['id']}, scene_id={r['scene_id']}, status={r['status']}, executed_at={r['executed_at']}")
        
if __name__ == "__main__":
    asyncio.run(main())
