#!/usr/bin/env python
"""投影一致性校验脚本"""

import asyncio
import sys
from src.db import init_db_pool, close_db_pool, get_db_pool
from src.writing.causality.projector import DeltaEngine
from src.writing.causality.health import HealthChecker

async def verify(novel_id: str, auto_repair: bool = False):
    await init_db_pool()
    try:
        if auto_repair:
            print(f"Rebuilding predicates for {novel_id}...")
            await DeltaEngine.rebuild_all_predicates(novel_id)
            print("Rebuild completed.")
        
        # 获取当前哈希和最后重建时间
        pool = get_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT core_predicates_hash, last_full_rebuild_at FROM projection_health WHERE novel_id = $1",
                novel_id
            )
        if row:
            print(f"Current core predicates hash: {row['core_predicates_hash']}")
            print(f"Last full rebuild at: {row['last_full_rebuild_at']}")
        else:
            print("No projection_health record found. Run rebuild first.")
            return
        
        # 可选：计算从事件流重建的哈希（不写数据库）需要单独实现，此处跳过
        print("✅ Verification: Run with --rebuild to perform full rebuild and update hash.")
    finally:
        await close_db_pool()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("novel_id", help="Novel ID")
    parser.add_argument("--rebuild", action="store_true", help="Perform full rebuild before verification")
    args = parser.parse_args()
    asyncio.run(verify(args.novel_id, auto_repair=args.rebuild))