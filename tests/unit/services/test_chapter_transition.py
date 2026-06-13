import pytest
from unittest.mock import AsyncMock, patch
from src.writing.services.chapter_transition import ChapterTransitionService, ChapterTransitionCommand
from src.orchestrator.state_patch import WorkflowPhase


class TestChapterTransitionService:
    @pytest.mark.asyncio
    async def test_execute_no_volume_finish(self):
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.transaction.return_value.__aenter__.return_value = None

        with patch("src.writing.services.chapter_transition.get_db_pool", return_value=mock_pool):
            cmd = ChapterTransitionCommand(
                novel_id="test", current_volume=1, current_chapter=5, total_chapters_in_volume=10, outline=None
            )
            result = await ChapterTransitionService.execute(cmd)

            assert result.volume_finished is False
            patch = result.state_patch
            assert patch.current_chapter == 6
            assert patch.current_volume == 1
            assert patch.current_scene_index == 0
            assert patch.scene_plan_list == []
            assert patch.total_scenes_in_chapter == 0
            assert patch.phase == WorkflowPhase.PLANNING
            # 验证数据库更新（只更新 chapter）
            mock_conn.execute.assert_called_once()
            args, _ = mock_conn.execute.call_args
            assert "SET current_chapter = $1" in args[0]

    @pytest.mark.asyncio
    async def test_execute_volume_finished_with_outline(self):
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.transaction.return_value.__aenter__.return_value = None

        outline = {"volumes": [{"chapters": [1,2,3,4,5,6,7,8,9,10]}, {"chapters": [1,2,3,4,5,6,7,8,9,10]}]}
        with patch("src.writing.services.chapter_transition.get_db_pool", return_value=mock_pool):
            cmd = ChapterTransitionCommand(
                novel_id="test", current_volume=1, current_chapter=10, total_chapters_in_volume=10, outline=outline
            )
            result = await ChapterTransitionService.execute(cmd)

            assert result.volume_finished is True
            patch = result.state_patch
            assert patch.current_chapter == 1
            assert patch.current_volume == 2
            assert patch.total_chapters_in_volume == 10
            assert patch.phase == WorkflowPhase.PLANNING
            # 验证数据库更新（volume + chapter）
            mock_conn.execute.assert_called_once()
            args, _ = mock_conn.execute.call_args
            assert "SET current_volume = $1, current_chapter = $2" in args[0]