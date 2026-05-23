import pytest
import json
from pathlib import Path
from src.db import init_db_pool, close_db_pool, get_db_pool
from src.writing.event_store import NarrativeEventStore
from src.writing.snapshot import SnapshotManager
from src.writing.delta import StateDelta
from src.writing.world_state import WorldState

GOLDEN_FILE = Path(__file__).parent.parent / "golden_replay" / "golden.json"


@pytest.mark.asyncio
async def test_deterministic_replay():
    if not GOLDEN_FILE.exists():
        pytest.skip(f"Golden file not found: {GOLDEN_FILE}. Run scripts/generate_golden.py first.")

    await init_db_pool()
    pool = get_db_pool()
    try:
        with open(GOLDEN_FILE) as f:
            golden = json.load(f)

        novel_id = golden["novel_id"]
        event_store = NarrativeEventStore(pool)
        snap_mgr = SnapshotManager(pool)

        for checkpoint in golden["checkpoints"]:
            event_id = checkpoint["event_id"]
            expected_world_hash = checkpoint["world_hash"]

            world, _, last_id = await snap_mgr.load_latest_snapshot(novel_id)
            if world is None:
                world = WorldState()
                last_id = 0

            events = await event_store.get_events_since(novel_id, last_id, limit=10000)
            for evt_id, evt in events:
                if evt_id > event_id:
                    break
                world = world.apply_delta(StateDelta(events=[evt]))

            actual_hash = world.get_state_hash()
            assert actual_hash == expected_world_hash, \
                f"Mismatch at event_id {event_id} (vol {checkpoint['volume']} ch {checkpoint['chapter']})"
    finally:
        await close_db_pool()