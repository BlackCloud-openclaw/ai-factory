# tests/phase13/test_planner_output.py
import pytest
from src.writing.narrative_intent import NarrativeIntent, SceneRole
from src.writing.planner_output import PlannerOutput
from src.writing.planning_contract import PlanningContract, Intent, ContractMetadata, Execution, ExecutionUnit

def test_planner_output_creation():
    intent = NarrativeIntent(
        intent_id="test",
        scene_role=SceneRole.CONFLICT_ESCALATION,
        objective="测试叙事意图"  # 5个字符 ✅
    )
    contract = PlanningContract(
        scene_id="test",
        intent=Intent(
            goal="测试目标",        # 至少3个字符
            conflict="测试冲突",    # 至少3个字符
            expected_outcome="测试结果"  # 至少3个字符
        ),
        execution=Execution(units=[]),
        metadata=ContractMetadata(chapter=1, scene_index=0)
    )
    output = PlannerOutput(
        narrative_intent=intent,
        execution_contract=contract
    )
    assert output.narrative_intent.scene_role == SceneRole.CONFLICT_ESCALATION
    assert output.execution_contract.intent.goal == "测试目标"