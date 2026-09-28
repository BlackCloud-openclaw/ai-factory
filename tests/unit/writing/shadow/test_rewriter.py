"""
Phase 15.3 — ShadowRewriter 单元测试
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.writing.shadow.rewriter import ShadowRewriter
from src.writing.shadow.prompt_builder import ShadowPromptBuilder
from src.writing.shadow.llm_client import LLMClient


class TestShadowRewriter:
    """ShadowRewriter 单元测试"""

    @pytest.fixture
    def mock_llm_client(self):
        mock = AsyncMock(spec=LLMClient)
        mock.generate.return_value = "rewritten text"
        return mock

    @pytest.fixture
    def mock_prompt_builder(self):
        mock = MagicMock(spec=ShadowPromptBuilder)
        mock.build.return_value = "Rewrite prompt"
        return mock

    @pytest.fixture
    def rewriter(self, mock_llm_client, mock_prompt_builder):
        return ShadowRewriter(mock_llm_client, mock_prompt_builder)

    @pytest.fixture
    def sample_contract(self):
        return {"scene_id": "test_scene"}

    async def test_rewrite_calls_prompt_builder(self, rewriter, mock_prompt_builder, sample_contract):
        """验证 rewrite 调用 prompt_builder.build"""
        await rewriter.rewrite("original text", sample_contract)
        mock_prompt_builder.build.assert_called_once_with("original text", sample_contract)

    async def test_rewrite_calls_llm_client(self, rewriter, mock_llm_client, sample_contract):
        """验证 rewrite 调用 llm_client.generate"""
        await rewriter.rewrite("original text", sample_contract)
        mock_llm_client.generate.assert_called_once_with("Rewrite prompt")

    async def test_rewrite_returns_llm_output(self, rewriter, sample_contract):
        """验证 rewrite 返回 LLM 输出"""
        result = await rewriter.rewrite("original text", sample_contract)
        assert result == "rewritten text"

    async def test_rewrite_propagates_llm_exception(self, rewriter, mock_llm_client, sample_contract):
        """验证 rewrite 传播 LLM 异常"""
        mock_llm_client.generate.side_effect = Exception("LLM error")
        with pytest.raises(Exception, match="LLM error"):
            await rewriter.rewrite("original text", sample_contract)