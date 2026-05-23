import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.writing.services.scene_completion import SceneCompletionService
from src.writing.services.models import SceneCompletionCommand
from src.orchestrator.state_patch import StatePatch, WorkflowPhase


class TestSceneCompletionService:
    @pytest.mark.asyncio
    async def test_execute_no_db_pool(self):
        with patch("src.writing.services.scene_completion.get_db_pool", return_value=None):
            cmd = SceneCompletionCommand(
                novel_id="test", volume=1, chapter=1, scene_idx=0, total_scenes=3,
                current_world_state={}, parsed_output={}, scene_plan=None
            )
            result = await SceneCompletionService.execute(cmd)
            assert result.error == "No db pool"
            assert result.state_patch.error == "Database pool unavailable"

    @pytest.mark.asyncio
    async def test_execute_with_events_and_chapter_finished(self):
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.transaction.return_value.__aenter__.return_value = None

        with patch("src.writing.services.scene_completion.get_db_pool", return_value=mock_pool):
            # 模拟事件解析和应用
            with patch("src.writing.services.scene_completion.event_from_dict") as mock_event_from_dict:
                mock_event = MagicMock()
                mock_event_from_dict.return_value = mock_event

                cmd = SceneCompletionCommand(
                    novel_id="test", volume=1, chapter=1, scene_idx=2, total_scenes=3,
                    current_world_state={"characters": {}}, 
                    parsed_output={"events": [{"type": "realm_upgrade", "actor": "LinYi"}]},
                    scene_plan=None
                )
                result = await SceneCompletionService.execute(cmd)

                # 验证数据库操作被调用（至少一次）
                assert mock_conn.execute.call_count >= 2  # UPDATE scene_execution_units + INSERT writing_progress
                assert result.chapter_finished is True
                assert result.state_patch.phase == WorkflowPhase.TRANSITIONING
                assert result.state_patch.current_scene_index == 3

    @pytest.mark.asyncio
    async def test_execute_no_events(self):
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.transaction.return_value.__aenter__.return_value = None

        with patch("src.writing.services.scene_completion.get_db_pool", return_value=mock_pool):
            cmd = SceneCompletionCommand(
                novel_id="test", volume=1, chapter=1, scene_idx=0, total_scenes=3,
                current_world_state={}, parsed_output={}, scene_plan=None
            )
            result = await SceneCompletionService.execute(cmd)

            assert result.events_applied == 0
            assert result.chapter_finished is False
            assert result.state_patch.phase == WorkflowPhase.WRITING
            assert result.state_patch.current_scene_index == 1