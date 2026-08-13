# tests/phase13/test_writing_contract_immutable.py
import pytest
from src.writing.contracts import WritingContract, WritingConstraints
from src.writing.scene_execution_context import SceneExecutionContext
from src.writing.narrative_intent import NarrativeIntent, SceneRole


def test_contract_immutable():
    ctx = SceneExecutionContext(
        chapter_id="c1",
        scene_id="s1",
        scene_role="setup",
        dramatic_function="intro",
    )
    contract = WritingContract(scene_context=ctx)
    with pytest.raises(AttributeError):
        contract.narrative_intent = "something"


def test_constraints_default_factory():
    constraints = WritingConstraints()
    assert constraints.must_events == []
    assert constraints.forbidden_events == []


def test_contract_with_intent():
    intent = NarrativeIntent(
        intent_id="test",
        scene_role=SceneRole.SETUP,
        objective="测试目标足够长",  # 至少5个字符
    )
    ctx = SceneExecutionContext(
        chapter_id="c1",
        scene_id="s1",
        scene_role="setup",
        dramatic_function="intro",
    )
    contract = WritingContract(
        scene_context=ctx,
        narrative_intent=intent,
    )
    assert contract.narrative_intent == intent