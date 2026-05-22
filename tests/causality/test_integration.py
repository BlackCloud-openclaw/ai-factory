import asyncio
import sys
import uuid
sys.path.insert(0, '/home/data/projects/ai_factory')

from src.db import init_db_pool, get_db_pool
from src.writing.event_store import NarrativeEventStore
from src.writing.events import ItemAcquireEvent

async def main():
    print("Initializing database pool...")
    await init_db_pool()
    pool = get_db_pool()
    if not pool:
        print("❌ Database pool not initialized")
        return

    store = NarrativeEventStore(pool)
    
    # 确保测试小说记录存在
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO novels (novel_id, title) 
            VALUES ('test_novel_001', 'Test Novel') 
            ON CONFLICT (novel_id) DO NOTHING
        """)
        print("✅ Novel record ready")

    # 创建测试事件
    event = ItemAcquireEvent(
        event_id=str(uuid.uuid4()),  # 生成有效 UUID
        actor="LinYi",
        item="Sword",
        source="chest"
    )
    
    print("Appending event...")
    await store.append_event("test_novel_001", event, volume_num=1, chapter_num=1)
    print("✅ Event appended")
    
    # 验证谓词表
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM predicates WHERE novel_id = 'test_novel_001' AND subject = 'LinYi' AND relation = 'has_item'"
        )
        if row:
            print(f"✅ Predicate found: subject={row['subject']}, relation={row['relation']}, object={row['object']}")
        else:
            print("❌ No predicate found")

if __name__ == "__main__":
    asyncio.run(main())