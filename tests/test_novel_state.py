import pytest
import asyncio
import asyncpg
from src.writing.events import Event, EVENT_CHARACTER_UPDATE
from src.writing.reducer import apply_event, reduce_state
from src.writing.invariants import validate_event
from src.writing.event_store import EventStore

# 需要提供测试数据库连接
TEST_DSN = "postgresql://woami:kali@localhost:5432/ai_factory"

@pytest.fixture
async def event_store():
    pool = await asyncpg.create_pool(TEST_DSN, min_size=1, max_size=1)
    store = EventStore(pool)
    yield store
    await pool.close()

@pytest.mark.asyncio
async def test_insert_and_load_events(event_store):
    # 创建一个事件
    event = Event.new(
        event_type=EVENT_CHARACTER_UPDATE,
        payload={"name": "林风", "updates": {"realm": "筑基"}},
        novel_id="test_novel"
    )
    seq = await event_store.insert_event(event)
    assert seq > 0
    
    # 加载事件
    events = await event_store.load_events("test_novel")
    assert len(events) == 1
    assert events[0].type == EVENT_CHARACTER_UPDATE
    assert events[0].payload["name"] == "林风"

def test_apply_event():
    state = {}
    event = Event.new(EVENT_CHARACTER_UPDATE, {"name": "林风", "updates": {"realm": "筑基"}}, "test")
    new_state = apply_event(state, event)
    assert new_state["characters"]["林风"]["realm"] == "筑基"

def test_reduce_state():
    events = [
        Event.new(EVENT_CHARACTER_UPDATE, {"name": "林风", "updates": {"realm": "炼气"}}, "test"),
        Event.new(EVENT_CHARACTER_UPDATE, {"name": "林风", "updates": {"realm": "筑基"}}, "test"),
    ]
    # 模拟 sequence_id
    for i, e in enumerate(events, 1):
        e.sequence_id = i
    state = reduce_state(events)
    assert state["characters"]["林风"]["realm"] == "筑基"

def test_validate_event_realm_skip():
    state = {"characters": {"林风": {"realm": "筑基"}}}
    event = Event.new(EVENT_CHARACTER_UPDATE, {"name": "林风", "updates": {"realm": "元婴"}}, "test")
    ok, msg = validate_event(state, event)
    assert not ok
    assert "跨级" in msg

@pytest.mark.asyncio
async def test_snapshot_save_load(event_store):
    # 保存快照
    state = {"characters": {"林风": {"realm": "筑基"}}}
    await event_store.save_snapshot("test_novel", state, last_sequence_id=5)
    
    # 加载快照
    loaded_state, last_seq = await event_store.load_snapshot("test_novel")
    assert loaded_state["characters"]["林风"]["realm"] == "筑基"
    assert last_seq == 5