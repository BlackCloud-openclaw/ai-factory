import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.writing.services.scene_planning import ScenePlanningService
from src.writing.services.models import ScenePlanningCommand
from src.orchestrator.state_patch import WorkflowPhase


class TestScenePlanningService:
    @pytest.mark.asyncio
    async def test_execute_with_scenes(self):
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        with patch("src.writing.services.scene_planning.get_db_pool", return_value=mock_pool), \
             patch("src.writing.services.scene_planning.PlannerAgent") as MockPlanner, \
             patch("src.writing.services.scene_planning.WorldState.from_dict", return_value=MagicMock()), \
             patch("src.writing.services.scene_planning.ensure_core_predicates", AsyncMock()), \
             patch("src.writing.services.scene_planning.ContextCompiler") as MockCompiler:

            mock_compiler = MockCompiler.return_value
            mock_compiler.compile_for_planner.return_value = "compiled"
            mock_planner = MockPlanner.return_value
            mock_planner.run = AsyncMock(return_value={"scene_plan": {"scenes": [{"goal": "g1"}]}})

            cmd = ScenePlanningCommand(
                novel_id="test", volume=1, chapter=1, task_type="scene_plan",
                outline={"volumes": [{"chapters": [{"must_events": []}]}]},
                current_state={}, user_input="test", resume=False, total_chapters_in_volume=10
            )
            result = await ScenePlanningService.execute(cmd)

            assert result.error is None
            assert result.total_scenes == 1
            patch = result.state_patch
            assert patch.phase == WorkflowPhase.WRITING
            assert patch.scene_plan_list == [{"goal": "g1"}]
            assert patch.total_scenes_in_chapter == 1
            assert patch.current_scene_index == 0

    @pytest.mark.asyncio
    async def test_execute_no_scene_plan(self):
        with patch("src.writing.services.scene_planning.get_db_pool", return_value=AsyncMock()), \
             patch("src.writing.services.scene_planning.PlannerAgent") as MockPlanner:

            mock_planner = MockPlanner.return_value
            mock_planner.run = AsyncMock(return_value={})  # 无 scene_plan
            cmd = ScenePlanningCommand(novel_id="test", volume=1, chapter=1, task_type="scene_plan",
                                       outline={}, current_state={}, user_input="test", resume=False)
            result = await ScenePlanningService.execute(cmd)
            assert result.error == "No scene plan generated"
            assert result.state_patch.error == "No scene plan generated"