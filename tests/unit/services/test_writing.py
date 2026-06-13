import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.writing.services.writing import WritingService
from src.writing.services.models import WritingCommand


class TestWritingService:
    @pytest.mark.asyncio
    async def test_execute_with_valid_json(self):
        with patch("src.writing.services.writing.WritingAgent") as MockWriter, \
             patch("src.writing.services.writing.VoiceprintRegistry"), \
             patch("src.writing.services.writing.ContextCompiler"), \
             patch("src.writing.services.writing.WorldState.from_dict"):

            mock_writer = MockWriter.return_value
            mock_writer.run = AsyncMock(return_value={
                "scene_text": '{"scene_text": "hello world", "events": [{"type": "discovery", "importance": 5}]}'
            })

            cmd = WritingCommand(
                novel_id="test", volume=1, chapter=1, scene_idx=0,
                scene_plan={"goal": "test"}, current_state={}, writing_feedback=""
            )
            result = await WritingService.execute(cmd)

            assert result.error is None
            assert result.scene_text == "hello world"
            # 验证 importance 转换
            assert result.events[0]["importance"] == "critical"
            assert result.state_patch.final_answer == "hello world"

    @pytest.mark.asyncio
    async def test_execute_no_output(self):
        with patch("src.writing.services.writing.WritingAgent") as MockWriter, \
             patch("src.writing.services.writing.VoiceprintRegistry"), \
             patch("src.writing.services.writing.ContextCompiler"):

            mock_writer = MockWriter.return_value
            mock_writer.run = AsyncMock(return_value={"scene_text": None})

            cmd = WritingCommand(
                novel_id="test", volume=1, chapter=1, scene_idx=0,
                scene_plan={}, current_state={}, writing_feedback=""
            )
            result = await WritingService.execute(cmd)
            assert result.error == "No output from writer"
            assert result.state_patch.error == "No output from writer"