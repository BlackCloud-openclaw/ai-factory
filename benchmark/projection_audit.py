#!/usr/bin/env python
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db import init_db_pool, close_db_pool, get_db_pool
from src.writing.causality.health import HealthChecker

async def audit_projection(novel_id: str):
    await init_db_pool()
    pool = get_db_pool()
    
    async with pool.acquire() as conn:
        # 获取当前核心谓词哈希
        current_hash = await HealthChecker._compute_core_predicates_hash(novel_id)
        
        # 获取死信数量
        dead_count = await conn.fetchval(
            "SELECT COUNT(*) FROM projection_dead_letters WHERE novel_id = $1", novel_id
        )
        
        # 获取投影滞后
        last_event_id = await conn.fetchval(
            "SELECT MAX(id) FROM narrative_events WHERE novel_id = $1", novel_id
        )
        last_projected = await conn.fetchval(
            "SELECT last_projected_event_id FROM projection_health WHERE novel_id = $1", novel_id
        )
        lag = (last_event_id or 0) - (last_projected or 0)
        
        # 检查林逸是否有 is_alive 和 realm 谓词
        is_alive_exists = await conn.fetchval(
            "SELECT 1 FROM predicates WHERE novel_id=$1 AND subject='林逸' AND relation='is_alive' AND is_active=true",
            novel_id
        )
        realm_exists = await conn.fetchval(
            "SELECT 1 FROM predicates WHERE novel_id=$1 AND subject='林逸' AND relation='realm' AND is_active=true",
            novel_id
        )
    
    await close_db_pool()
    
    result = {
        "projection_match": (is_alive_exists and realm_exists and dead_count == 0 and lag < 100),
        "current_hash": current_hash,
        "dead_letter_count": dead_count,
        "projection_lag": lag,
        "protagonist_is_alive": bool(is_alive_exists),
        "protagonist_realm_exists": bool(realm_exists),
        "message": "投影审计完成"
    }
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    novel_id = sys.argv[1] if len(sys.argv) > 1 else "simple_long_novel_001"
    asyncio.run(audit_projection(novel_id))
