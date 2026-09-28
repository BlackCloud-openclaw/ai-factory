"""
Phase 15.3 — Shadow Recorder 单元测试
"""

import pytest
from unittest.mock import AsyncMock, Mock

from src.writing.shadow.result import ShadowRewriteResult, ShadowRewriteStatus
from src.writing.shadow.recorder import MemoryShadowRecorder


class TestMemoryShadowRecorder:
    """MemoryShadowRecorder 单元测试"""

    @pytest.fixture
    def recorder(self):
        return MemoryShadowRecorder()

    @pytest.fixture
    def sample_result(self):
        return ShadowRewriteResult(
            scene_id="scene_001",
            original_text="original",
            original_validation_passed=True,
            original_violations=[],
            rewritten_text="rewritten",
            rewritten_validation_passed=True,
            rewritten_violations=[],
            status=ShadowRewriteStatus.SUCCESS,
            experiment_id="exp_001",
            contract_id="contract_001",
            model="Qwen3-32B",
        )

    @pytest.fixture
    def failed_result(self):
        return ShadowRewriteResult(
            scene_id="scene_002",
            original_text="original",
            original_validation_passed=True,
            original_violations=[],
            rewritten_text="bad rewritten",
            rewritten_validation_passed=False,
            rewritten_violations=["missing_plot_flag"],
            status=ShadowRewriteStatus.VALIDATION_FAILED,
            experiment_id="exp_001",
            contract_id="contract_001",
            model="Qwen3-32B",
        )

    async def test_record_and_retrieve(self, recorder, sample_result):
        await recorder.record(sample_result)
        results = await recorder.get_by_scene("scene_001")
        assert len(results) == 1
        assert results[0].scene_id == "scene_001"
        assert results[0].status == ShadowRewriteStatus.SUCCESS
        assert results[0].contract_id == "contract_001"
        assert results[0].model == "Qwen3-32B"

    async def test_multiple_records(self, recorder, sample_result, failed_result):
        await recorder.record(sample_result)
        await recorder.record(failed_result)

        results = await recorder.get_by_experiment("exp_001")
        assert len(results) == 2

        scene_001_results = await recorder.get_by_scene("scene_001")
        assert len(scene_001_results) == 1

        scene_002_results = await recorder.get_by_scene("scene_002")
        assert len(scene_002_results) == 1
        assert scene_002_results[0].status == ShadowRewriteStatus.VALIDATION_FAILED

    async def test_count_by_status(self, recorder, sample_result, failed_result):
        await recorder.record(sample_result)
        await recorder.record(failed_result)

        success_count = await recorder.count_by_status("success")
        assert success_count == 1

        fail_count = await recorder.count_by_status("validation_failed")
        assert fail_count == 1

    async def test_limit_correctness(self, recorder):
        """测试 limit 正确性：同一 scene 多条记录，limit 生效"""
        for i in range(15):
            result = ShadowRewriteResult(
                scene_id="scene_001",
                original_text="original",
                original_validation_passed=True,
                original_violations=[],
                rewritten_text=f"rewritten_{i}",
                rewritten_validation_passed=True,
                rewritten_violations=[],
                status=ShadowRewriteStatus.SUCCESS,
            )
            await recorder.record(result)

        results = await recorder.get_by_scene("scene_001", limit=5)
        assert len(results) == 5

    async def test_get_by_scene_unknown(self, recorder):
        results = await recorder.get_by_scene("unknown")
        assert len(results) == 0

    async def test_get_by_experiment_unknown(self, recorder):
        results = await recorder.get_by_experiment("unknown")
        assert len(results) == 0

    async def test_all(self, recorder, sample_result, failed_result):
        await recorder.record(sample_result)
        await recorder.record(failed_result)

        all_records = await recorder.all()
        assert len(all_records) == 2