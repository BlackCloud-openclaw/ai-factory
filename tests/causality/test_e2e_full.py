import pytest
import uuid
from src.writing.event_store import NarrativeEventStore
from src.writing.events import ItemAcquireEvent

@pytest.mark.skip(reason="E2E requires full LLM and workflow, skip for CI")
@pytest.mark.asyncio
async def test_outline_generation(db_pool, novel_id):
    pass

@pytest.mark.skip(reason="E2E requires full LLM and workflow, skip for CI")
@pytest.mark.asyncio
async def test_scene_plan_and_writing(db_pool, novel_id):
    pass

@pytest.mark.skip(reason="E2E requires full LLM and workflow, skip for CI")
@pytest.mark.asyncio
async def test_consistency_budget(db_pool, novel_id):
    pass

@pytest.mark.skip(reason="Database init issue with test database, skip for now")
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
