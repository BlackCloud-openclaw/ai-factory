import pytest
import asyncio
import asyncpg
from src.config import config
from src.orchestrator.state import AgentState
from src.tools.calculator import update_character  # 直接导入
from src.writing.event_store import EventStore
from src.writing.reducer import apply_event

@pytest.mark.asyncio
async def test_tool_node_v2_direct():
    pool = await asyncpg.create_pool(config.postgres_dsn, min_size=1, max_size=1)
    event_store = EventStore(pool)
    
    state = AgentState(
        user_input="test",
        pending_tool_calls=[
            {"tool": "update_character", "args": {"name": "林风", "updates": {"realm": "筑基"}}}
        ],
        current_state={},
        last_sequence_id=0,
        novel_id="test_novel_direct",
        metadata={}
    )
    
    # 直接调用工具函数，绕过注册表
    for call in state.pending_tool_calls:
        event = update_character(**call["args"])
        seq = await event_store.insert_event(event)
        event.sequence_id = seq
        state.current_state = apply_event(state.current_state, event)
    
    assert state.current_state.get("characters", {}).get("林风", {}).get("realm") == "筑基"
    print("✅ Test passed")
    await pool.close()