"""
Phase 15.3 — ShadowRewriteRunner 单元测试
"""

import pytest
from unittest.mock import AsyncMock, Mock

from src.writing.shadow import ShadowRewriteRunner, ShadowRewriteResult, ShadowRewriteStatus


class TestShadowRewriteRunner:
    """ShadowRewriteRunner 的行为契约测试"""

    @pytest.fixture
    def mock_rewriter(self):
        return AsyncMock()

    @pytest.fixture
    def mock_validator(self):
        return AsyncMock()

    @pytest.fixture
    def runner(self, mock_rewriter, mock_validator):
        return ShadowRewriteRunner(mock_rewriter, mock_validator)

    @pytest.fixture
    def sample_contract(self):
        return {"scene_id": "test_scene"}

    @pytest.fixture
    def original_validation_passed(self):
        return Mock(passed=True, violations=[])

    @pytest.fixture
    def original_validation_failed(self):
        return Mock(passed=False, violations=["missing_event"])

    @pytest.fixture
    def base_kwargs(self, sample_contract, original_validation_passed):
        return {
            "scene_id": "scene_001",
            "original_text": "original text",
            "contract": sample_contract,
            "original_validation_result": original_validation_passed,
        }

    async def test_success_path_original_pass_rewrite_pass(
        self, runner, mock_rewriter, mock_validator, base_kwargs
    ):
        """场景：Original PASS，Rewrite PASS → SUCCESS"""
        mock_rewriter.rewrite.return_value = "rewritten text"
        mock_validator.validate.return_value = Mock(passed=True, violations=[])

        result = await runner.submit(**base_kwargs)

        assert result.status == ShadowRewriteStatus.SUCCESS
        assert result.original_validation_passed is True
        assert result.rewritten_validation_passed is True
        assert result.scene_id == "scene_001"
        assert result.original_length == len("original text")
        assert result.rewritten_length == len("rewritten text")
        assert result.is_regression is False
        assert result.is_complete is True

    async def test_success_path_original_pass_rewrite_pass_with_experiment_id(
        self, runner, mock_rewriter, mock_validator, base_kwargs
    ):
        """场景：带 experiment_id 的 SUCCESS 路径"""
        mock_rewriter.rewrite.return_value = "rewritten text"
        mock_validator.validate.return_value = Mock(passed=True, violations=[])

        result = await runner.submit(**base_kwargs, experiment_id="exp_001")

        assert result.experiment_id == "exp_001"
        assert result.status == ShadowRewriteStatus.SUCCESS

    async def test_validation_failed_original_pass_rewrite_fail(
        self, runner, mock_rewriter, mock_validator, base_kwargs
    ):
        """场景：Original PASS，Rewrite FAIL → VALIDATION_FAILED"""
        mock_rewriter.rewrite.return_value = "bad rewritten text"
        mock_validator.validate.return_value = Mock(passed=False, violations=["missing_plot_flag"])

        result = await runner.submit(**base_kwargs)

        assert result.status == ShadowRewriteStatus.VALIDATION_FAILED
        assert result.original_validation_passed is True
        assert result.rewritten_validation_passed is False
        assert result.is_regression is True  # Original PASS → Rewrite FAIL
        assert result.is_complete is True
        assert result.rewritten_violations == ["missing_plot_flag"]

    async def test_llm_error(
        self, runner, mock_rewriter, mock_validator, base_kwargs
    ):
        """场景：Rewrite LLM 抛出异常 → LLM_ERROR"""
        mock_rewriter.rewrite.side_effect = Exception("Connection refused")

        result = await runner.submit(**base_kwargs)

        assert result.status == ShadowRewriteStatus.LLM_ERROR
        assert result.error_message == "Connection refused"
        assert result.rewritten_text is None
        assert result.rewritten_validation_passed is None
        assert result.is_complete is False
        assert result.is_regression is False

    async def test_validator_error(
        self, runner, mock_rewriter, mock_validator, base_kwargs
    ):
        """场景：Rewrite 成功，但 Validator 抛出异常 → VALIDATOR_ERROR"""
        mock_rewriter.rewrite.return_value = "rewritten text"
        mock_validator.validate.side_effect = Exception("Validator internal error")

        result = await runner.submit(**base_kwargs)

        assert result.status == ShadowRewriteStatus.VALIDATOR_ERROR
        assert "Validator" in result.error_message
        assert result.rewritten_text == "rewritten text"
        assert result.rewritten_validation_passed is None
        assert result.is_complete is False
        assert result.is_regression is False

    async def test_timeout(
        self, runner, mock_rewriter, mock_validator, base_kwargs
    ):
        """场景：Rewrite 超时 → TIMEOUT"""
        mock_rewriter.rewrite.side_effect = TimeoutError("Timeout after 30s")

        result = await runner.submit(**base_kwargs)

        assert result.status == ShadowRewriteStatus.TIMEOUT
        assert result.error_message == "Timeout after 30s"
        assert result.rewritten_text is None
        assert result.rewritten_validation_passed is None
        assert result.is_complete is False
        assert result.is_regression is False

    async def test_regression_not_triggered_on_llm_error(
        self, runner, mock_rewriter, mock_validator, base_kwargs
    ):
        """LLM_ERROR 不应触发 regression"""
        mock_rewriter.rewrite.side_effect = Exception("LLM connection failed")

        result = await runner.submit(**base_kwargs)

        assert result.is_regression is False
        assert result.status == ShadowRewriteStatus.LLM_ERROR

    async def test_regression_not_triggered_on_validator_error(
        self, runner, mock_rewriter, mock_validator, base_kwargs
    ):
        """VALIDATOR_ERROR 不应触发 regression"""
        mock_rewriter.rewrite.return_value = "rewritten text"
        mock_validator.validate.side_effect = Exception("Validator crashed")

        result = await runner.submit(**base_kwargs)

        assert result.is_regression is False
        assert result.status == ShadowRewriteStatus.VALIDATOR_ERROR

    async def test_regression_not_triggered_on_timeout(
        self, runner, mock_rewriter, mock_validator, base_kwargs
    ):
        """TIMEOUT 不应触发 regression"""
        mock_rewriter.rewrite.side_effect = TimeoutError("Timeout")

        result = await runner.submit(**base_kwargs)

        assert result.is_regression is False
        assert result.status == ShadowRewriteStatus.TIMEOUT

    async def test_regression_triggered_on_validation_failed(
        self, runner, mock_rewriter, mock_validator, base_kwargs
    ):
        """VALIDATION_FAILED 应触发 regression"""
        mock_rewriter.rewrite.return_value = "bad text"
        mock_validator.validate.return_value = Mock(passed=False, violations=["missing"])

        result = await runner.submit(**base_kwargs)

        assert result.is_regression is True
        assert result.status == ShadowRewriteStatus.VALIDATION_FAILED

    async def test_original_validation_result_used_as_reference(
        self, runner, mock_rewriter, mock_validator, sample_contract, original_validation_failed
    ):
        """场景：Original 生产阶段 FAIL，Rewrite PASS → SUCCESS（不重新验证 Original）"""
        mock_rewriter.rewrite.return_value = "good rewritten text"
        mock_validator.validate.return_value = Mock(passed=True, violations=[])

        result = await runner.submit(
            scene_id="scene_005",
            original_text="original text",
            contract=sample_contract,
            original_validation_result=original_validation_failed,
        )

        # 关键：Original 的验证结果直接来自传入的 original_validation_failed
        # Runner 不重新调用 Validator 验证 Original
        assert result.original_validation_passed is False
        assert result.original_violations == ["missing_event"]
        assert result.rewritten_validation_passed is True
        assert result.status == ShadowRewriteStatus.SUCCESS
        assert result.is_regression is False  # Original 已经是 FAIL，不算退化

    async def test_original_pass_rewrite_validation_failed_regression_true(
        self, runner, mock_rewriter, mock_validator, sample_contract, original_validation_passed
    ):
        """验证：Original PASS，Rewrite VALIDATION_FAILED → regression True"""
        mock_rewriter.rewrite.return_value = "bad text"
        mock_validator.validate.return_value = Mock(passed=False, violations=["missing"])

        result = await runner.submit(
            scene_id="scene_006",
            original_text="original text",
            contract=sample_contract,
            original_validation_result=original_validation_passed,
        )

        assert result.original_validation_passed is True
        assert result.rewritten_validation_passed is False
        assert result.is_regression is True
        assert result.status == ShadowRewriteStatus.VALIDATION_FAILED

    async def test_original_fail_rewrite_pass_regression_false(
        self, runner, mock_rewriter, mock_validator, sample_contract, original_validation_failed
    ):
        """验证：Original FAIL，Rewrite PASS → regression False"""
        mock_rewriter.rewrite.return_value = "good text"
        mock_validator.validate.return_value = Mock(passed=True, violations=[])

        result = await runner.submit(
            scene_id="scene_007",
            original_text="original text",
            contract=sample_contract,
            original_validation_result=original_validation_failed,
        )

        assert result.original_validation_passed is False
        assert result.rewritten_validation_passed is True
        assert result.is_regression is False
        assert result.status == ShadowRewriteStatus.SUCCESS

    async def test_original_fail_rewrite_fail_regression_false(
        self, runner, mock_rewriter, mock_validator, sample_contract, original_validation_failed
    ):
        """验证：Original FAIL，Rewrite FAIL → regression False（两边都 FAIL，不算退化）"""
        mock_rewriter.rewrite.return_value = "bad text"
        mock_validator.validate.return_value = Mock(passed=False, violations=["missing"])

        result = await runner.submit(
            scene_id="scene_008",
            original_text="original text",
            contract=sample_contract,
            original_validation_result=original_validation_failed,
        )

        assert result.original_validation_passed is False
        assert result.rewritten_validation_passed is False
        assert result.is_regression is False
        assert result.status == ShadowRewriteStatus.VALIDATION_FAILED