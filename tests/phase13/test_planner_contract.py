"""
Phase 13.1: Planner Output Contract 测试
"""

import pytest
from src.writing.narrative_intent import NarrativeIntent, SceneRole, NarrativeConsequence
from src.writing.planner_output import PlannerOutput
from src.writing.planning_contract import PlanningContract, Intent, ContractMetadata, Execution, ExecutionUnit


def test_planner_output_creation():
    """验证 PlannerOutput 可正常创建"""
    intent = NarrativeIntent(
        intent_id="test_intent",
        scene_role=SceneRole.CONFLICT_ESCALATION,
        objective="测试叙事意图"
    )

    contract = PlanningContract(
        scene_id="test_scene",
        intent=Intent(goal="测试目标", conflict="测试冲突", expected_outcome="测试结果"),
        execution=Execution(units=[]),
        metadata=ContractMetadata(chapter=1, scene_index=0)
    )

    output = PlannerOutput(
        narrative_intent=intent,
        execution_contract=contract
    )

    assert output.narrative_intent.scene_role == SceneRole.CONFLICT_ESCALATION
    assert output.execution_contract.scene_id == "test_scene"


def test_planner_output_serialization():
    """验证 PlannerOutput JSON 序列化"""
    intent = NarrativeIntent(
        intent_id="test_intent",
        scene_role=SceneRole.DISCOVERY,
        objective="发现隐藏线索",
        consequences=[
            NarrativeConsequence(
                target="knowledge.clue_found",
                operation="set",
                value=True,
                event_type="plot_flag_set"
            )
        ]
    )

    contract = PlanningContract(
        scene_id="test_scene",
        intent=Intent(goal="测试目标", conflict="测试冲突", expected_outcome="测试结果"),
        execution=Execution(units=[]),
        metadata=ContractMetadata(chapter=1, scene_index=0)
    )

    output = PlannerOutput(narrative_intent=intent, execution_contract=contract)
    data = output.model_dump()

    assert data["narrative_intent"]["scene_role"] == "discovery"
    assert data["execution_contract"]["scene_id"] == "test_scene"


def test_intent_id_deterministic():
    """验证 intent_id 确定性生成"""
    id1 = NarrativeIntent.generate_intent_id(
        scene_id="scene_1_58_0",
        role=SceneRole.CONFLICT_ESCALATION,
        objective="让主角意识到师门背后的真实目的"
    )
    id2 = NarrativeIntent.generate_intent_id(
        scene_id="scene_1_58_0",
        role=SceneRole.CONFLICT_ESCALATION,
        objective="让主角意识到师门背后的真实目的"
    )
    assert id1 == id2

    id3 = NarrativeIntent.generate_intent_id(
        scene_id="scene_1_58_1",
        role=SceneRole.CONFLICT_ESCALATION,
        objective="让主角意识到师门背后的真实目的"
    )
    assert id1 != id3