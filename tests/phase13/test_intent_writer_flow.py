# tests/phase13/test_intent_writer_flow.py
import pytest
from src.orchestrator.state import AgentState
from src.writing.contracts import WritingContract, WritingGoal
from src.writing.narrative_intent import NarrativeIntent, SceneRole
from src.writing.scene_execution_context import SceneExecutionContext
from src.writing.controlled_writer import ControlledWriter


@pytest.fixture
def minimal_writer():
    writer = ControlledWriter.__new__(ControlledWriter)
    writer.api_base = "http://localhost:8082"
    writer.model = "test-model"
    writer.max_retries_per_segment = 2
    writer.enable_fallback = True
    writer._runtime_services = None
    return writer


@pytest.mark.asyncio
async def test_narrative_intent_reaches_writer(minimal_writer):
    intent = NarrativeIntent(
        intent_id="test-intent",
        scene_role=SceneRole.SETUP,
        objective="测试目标场景执行",  # 至少5个字符
        beats=["事件1", "事件2"],
        consequences=[],
    )
    state = AgentState(narrative_intent=intent)

    ctx = SceneExecutionContext(
        chapter_id="c1",
        scene_id="s1",
        scene_role="setup",
        dramatic_function="intro",
    )
    contract = WritingContract(
        scene_context=ctx,
        narrative_intent=state.narrative_intent,
        writing_goal=WritingGoal(
            goal="测试目标",
            conflict="测试冲突",
            expected_outcome="测试结果",
        ),
    )

    assert contract.narrative_intent is not None
    assert contract.narrative_intent.intent_id == "test-intent"

    writer = minimal_writer
    prompt = writer._build_segment_prompt(
        writing_contract=contract,
        segment_units=[],
        segment_idx=0,
        total_segments=1,
        previous_text="",
        previous_events=[],
        current_state={},
    )
    assert "🎯 叙事意图约束" in prompt
    assert "测试目标场景执行" in prompt
    assert "事件1" in prompt