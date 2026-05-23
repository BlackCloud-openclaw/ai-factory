import pytest
import uuid
from src.writing.event_store import NarrativeEventStore
from src.writing.events import ItemAcquireEvent

@pytest.mark.asyncio
async def test_event_store_basic(db_pool, novel_id):
    store = NarrativeEventStore(db_pool)
    event = ItemAcquireEvent(
        event_id=str(uuid.uuid4()),
        actor="LinYi",
        item="Sword",
        source="test"
    )
    await store.append_event(novel_id, event, volume_num=1, chapter_num=1)
    events = await store.get_chapter_events(novel_id, 1, 1)
    assert len(events) == 1
    assert events[0].actor == "LinYi"