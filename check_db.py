import asyncio
import asyncpg
import sys
sys.path.insert(0, '/home/data/projects/ai_factory')
from src.config import config

async def check():
    conn = await asyncpg.connect(config.postgres_dsn)
    rows = await conn.fetch("""
        SELECT novel_id, volume_num, chapter_num, scene_idx, version_type, 
               LENGTH(scene_text) AS len
        FROM narrative_versions
        WHERE novel_id = 'simple_long_novel_001'
        ORDER BY chapter_num, scene_idx, version_type
        LIMIT 20
    """)
    print(f"找到 {len(rows)} 条记录")
    for row in rows:
        print(dict(row))
    await conn.close()

asyncio.run(check())
