# tests/phase14/test_contract_validation_integration.py
import pytest
from unittest.mock import AsyncMock, patch

from src.writing.services.scene_planning import ScenePlanningService
from src.writing.services.models import ScenePlanningCommand
from src.writing.planning_contract import (
    PlanningContract,
    Intent,
    Execution,
    ExecutionUnit,
    Observables,
    ContractMetadata,
    StateChange,
    SignalSource,
)
from src.writing.contracts.exceptions import ContractValidationError


def create_valid_state_change(sc_id="sc_001"):
    return StateChange(
        id=sc_id,
        type="knowledge_gain",
        source=SignalSource.INFERRED,
        confidence=0.95,
        name="test_knowledge",
        value=True,
    )


def create_test_contract(scene_id="test_scene", state_changes=None, units=None):
    if state_changes is None:
        state_changes = [create_valid_state_change()]
    if units is None:
        units = [
            ExecutionUnit(id="U1", label="action", description="获得玉佩")
        ]
    return PlanningContract(
        version="1.0",
        scene_id=scene_id,
        intent=Intent(goal="测试目标", conflict="测试冲突", expected_outcome="测试结果"),
        execution=Execution(units=units),
        observables=Observables(state_changes=state_changes),
        constraints=[],
        metadata=ContractMetadata(chapter=1, scene_index=0),
    )


def create_valid_scene_with_contract(contract: PlanningContract):
    """生成一个包含 planning_contract 的场景字典，并确保 must_events 有效"""
    return {
        "planning_contract": contract.model_dump(),
        # 为了绕过 scene_planning 中的默认填充，我们直接提供 must_events
        "must_events": [unit.description for unit in contract.execution.units],
        "goal": "测试目标",
        "conflict": "测试冲突",
        "outcome": "测试结果",
        "characters": ["林逸"],
        "scene_id": contract.scene_id,
    }


class TestContractValidationIntegration:
    @pytest.mark.asyncio
    async def test_valid_contract_passes(self):
        contract = create_test_contract()
        scene = create_valid_scene_with_contract(contract)
        scenes = [scene]

        mock_planner = AsyncMock()
        mock_planner.run.return_value = {
            "scene_plan": {"scenes": scenes},
            "planner_outputs": [],
        }

        with patch("src.writing.services.scene_planning.PlannerAgent", return_value=mock_planner):
            cmd = ScenePlanningCommand(
                novel_id="test_novel",
                volume=1,
                chapter=1,
                task_type="scene_plan",
                outline={},
                current_state={},
                user_input="test",
                resume=False,
                total_chapters_in_volume=10,
                metadata={},
                intent_resolver=None,
            )

            result = await ScenePlanningService.execute(cmd)
            assert result.error is None
            assert "contract_validation" in result.state_patch.metadata
            assert result.state_patch.metadata["contract_validation"]["valid"] == 1

    @pytest.mark.asyncio
    async def test_contract_without_state_changes_fails(self):
        invalid_contract = create_test_contract(state_changes=[])
        scene = create_valid_scene_with_contract(invalid_contract)
        scenes = [scene]

        mock_planner = AsyncMock()
        mock_planner.run.return_value = {
            "scene_plan": {"scenes": scenes},
            "planner_outputs": [],
        }

        with patch("src.writing.services.scene_planning.PlannerAgent", return_value=mock_planner):
            cmd = ScenePlanningCommand(
                novel_id="test_novel",
                volume=1,
                chapter=1,
                task_type="scene_plan",
                outline={},
                current_state={},
                user_input="test",
                resume=False,
                total_chapters_in_volume=10,
                metadata={},
                intent_resolver=None,
            )

            with pytest.raises(ContractValidationError) as exc_info:
                await ScenePlanningService.execute(cmd)

            # ✅ 修改断言以匹配中文错误消息
            assert "state_changes" in exc_info.value.message.lower()

    @pytest.mark.asyncio
    async def test_contract_with_invalid_state_change_fails(self):
        invalid_sc = StateChange(
            id="invalid_001",
            type="invalid_type",
            source=SignalSource.INFERRED,
            confidence=0.5,
        )
        invalid_contract = create_test_contract(state_changes=[invalid_sc])
        scene = create_valid_scene_with_contract(invalid_contract)
        scenes = [scene]

        mock_planner = AsyncMock()
        mock_planner.run.return_value = {
            "scene_plan": {"scenes": scenes},
            "planner_outputs": [],
        }

        with patch("src.writing.services.scene_planning.PlannerAgent", return_value=mock_planner):
            cmd = ScenePlanningCommand(
                novel_id="test_novel",
                volume=1,
                chapter=1,
                task_type="scene_plan",
                outline={},
                current_state={},
                user_input="test",
                resume=False,
                total_chapters_in_volume=10,
                metadata={},
                intent_resolver=None,
            )

            with pytest.raises(ContractValidationError) as exc_info:
                await ScenePlanningService.execute(cmd)

            assert "Invalid StateChange.type" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_contract_with_warning_passes_with_warning_log(self):
        low_conf_sc = create_valid_state_change()
        low_conf_sc.confidence = 0.3
        contract = create_test_contract(state_changes=[low_conf_sc])
        scene = create_valid_scene_with_contract(contract)
        scenes = [scene]

        mock_planner = AsyncMock()
        mock_planner.run.return_value = {
            "scene_plan": {"scenes": scenes},
            "planner_outputs": [],
        }

        with patch("src.writing.services.scene_planning.PlannerAgent", return_value=mock_planner):
            cmd = ScenePlanningCommand(
                novel_id="test_novel",
                volume=1,
                chapter=1,
                task_type="scene_plan",
                outline={},
                current_state={},
                user_input="test",
                resume=False,
                total_chapters_in_volume=10,
                metadata={},
                intent_resolver=None,
            )

            result = await ScenePlanningService.execute(cmd)
            assert result.error is None

    @pytest.mark.asyncio
    async def test_contract_with_duplicate_state_change_ids_fails(self):
        sc1 = create_valid_state_change(sc_id="dup_id")
        sc2 = create_valid_state_change(sc_id="dup_id")
        contract = create_test_contract(state_changes=[sc1, sc2])
        scene = create_valid_scene_with_contract(contract)
        scenes = [scene]

        mock_planner = AsyncMock()
        mock_planner.run.return_value = {
            "scene_plan": {"scenes": scenes},
            "planner_outputs": [],
        }

        with patch("src.writing.services.scene_planning.PlannerAgent", return_value=mock_planner):
            cmd = ScenePlanningCommand(
                novel_id="test_novel",
                volume=1,
                chapter=1,
                task_type="scene_plan",
                outline={},
                current_state={},
                user_input="test",
                resume=False,
                total_chapters_in_volume=10,
                metadata={},
                intent_resolver=None,
            )

            with pytest.raises(ContractValidationError) as exc_info:
                await ScenePlanningService.execute(cmd)

            assert "duplicate" in exc_info.value.message.lower()