import pytest
import pytest_asyncio
import asyncpg
from src.writing.events import Event, EVENT_CHARACTER_UPDATE
from src.writing.event_store import EventStore
from src.config import config
import uuid

DB_DSN = config.postgres_dsn

@pytest_asyncio.fixture
async def event_store():
    pool = await asyncpg.create_pool(DB_DSN, min_size=1, max_size=1)
    store = EventStore(pool)
    yield store
    await pool.close()

@pytest.mark.asyncio
async def test_insert_and_load_event(event_store):
    novel_id = f"test_novel_{uuid.uuid4().hex[:8]}"
    
    # 删除该 novel_id 的旧事件（避免重复）
    async with event_store.pool.acquire() as conn:
        await conn.execute("DELETE FROM events WHERE novel_id = $1", novel_id)
    
    event = Event.new(
        event_type=EVENT_CHARACTER_UPDATE,
        payload={"name": "林风", "updates": {"realm": "筑基"}},
        novel_id=novel_id
    )
    seq = await event_store.insert_event(event)
    assert seq > 0

    events = await event_store.load_events(novel_id)
    assert len(events) == 1
    assert events[0].type == EVENT_CHARACTER_UPDATE
    assert events[0].payload["name"] == "林风"
    assert events[0].payload["updates"]["realm"] == "筑基"

@pytest.mark.asyncio
async def test_snapshot_save_and_load(event_store):
    novel_id = f"test_snap_{uuid.uuid4().hex[:8]}"
    
    # 清理旧快照
    async with event_store.pool.acquire() as conn:
        await conn.execute("DELETE FROM novels WHERE novel_id = $1", novel_id)
    
    state = {"characters": {"林风": {"realm": "筑基", "level": 2}}}
    await event_store.save_snapshot(novel_id, state, last_sequence_id=10)
    
    loaded_state, last_seq = await event_store.load_snapshot(novel_id)
    assert loaded_state is not None
    assert "characters" in loaded_state
    assert loaded_state["characters"]["林风"]["realm"] == "筑基"
    assert last_seq == 10