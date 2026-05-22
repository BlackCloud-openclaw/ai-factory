#!/usr/bin/env python
"""
直接测试 CausalityValidator，避免循环导入。
"""
import asyncio
import sys
import uuid
sys.path.insert(0, '/home/data/projects/ai_factory')

from src.db import init_db_pool, get_db_pool
from src.writing.causality.validator import CausalityValidator
from src.writing.causality.predicate import Predicate
from src.writing.event_store import NarrativeEventStore
from src.writing.events import ItemAcquireEvent

from src.writing.causality.rule_engine import RuleEngine
engine = RuleEngine()
print("Rules:", engine.rules)
print("Index:", engine._index_by_event_type)

async def setup_test_data(novel_id):
    pool = get_db_pool()
    if not pool:
        await init_db_pool()
        pool = get_db_pool()
    
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO novels (novel_id, title) VALUES ($1, $2) ON CONFLICT (novel_id) DO NOTHING",
            novel_id, "Test Novel"
        )
        await conn.execute("DELETE FROM predicates WHERE novel_id = $1", novel_id)
        
        event = ItemAcquireEvent(
            event_id=str(uuid.uuid4()),
            actor="LinYi",
            item="Sword",
            source="test"
        )
        store = NarrativeEventStore(pool)
        await store.append_event(novel_id, event, volume_num=1, chapter_num=1)
        print("✅ Inserted item_acquire event -> has_item(LinYi, Sword)")

async def load_predicates(novel_id):
    store = NarrativeEventStore(get_db_pool())
    return await store._load_active_predicates(novel_id)

async def test():
    novel_id = "test_causality_direct_001"
    print("Initializing DB...")
    await init_db_pool()
    await setup_test_data(novel_id)
    
    validator = CausalityValidator()
    predicates = await load_predicates(novel_id)
    print("Loaded predicates:")
    for key, pred in predicates.items():
        print(f"  {key}: {pred.subject} {pred.relation} {pred.object} (type={type(pred.object)})")    
    
    # Test 1: use owned item
    event_good = {"type": "use_item", "actor": "LinYi", "item": "Sword"}
    result1 = validator.validate(event_good, predicates)
    print(f"Owned -> passed={result1['passed']}, severity={result1['severity']}")
    assert result1["passed"] is True
    
    # Test 2: use unowned item
    event_bad = {"type": "use_item", "actor": "LinYi", "item": "MysticSword"}
    result2 = validator.validate(event_bad, predicates)
    print(f"Unowned -> passed={result2['passed']}, severity={result2['severity']}")
    assert result2["passed"] is False
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    asyncio.run(test())