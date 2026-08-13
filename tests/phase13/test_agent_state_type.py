# tests/phase13/test_agent_state_type.py
import pytest
from src.orchestrator.state import AgentState
from src.writing.narrative_intent import NarrativeIntent, SceneRole


def test_agent_state_narrative_intent_type():
    intent = NarrativeIntent(
        intent_id="test",
        scene_role=SceneRole.SETUP,
        objective="测试目标足够长",  # 至少5个字符
    )
    state = AgentState()
    state.narrative_intent = intent
    assert isinstance(state.narrative_intent, NarrativeIntent)