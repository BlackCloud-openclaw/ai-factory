"""
Phase 15.3 — Shadow Rewriter

Rewriter 实现：使用 LLMClient 执行 Shadow Rewrite。
"""

from typing import Any
from .runner import Rewriter
from .prompt_builder import ShadowPromptBuilder
from .llm_client import LLMClient


class ShadowRewriter(Rewriter):
    """
    使用独立 LLMClient 执行 Rewrite。

    不依赖 ControlledWriter，不修改任何生产状态。
    """

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_builder: ShadowPromptBuilder,
    ):
        self._llm_client = llm_client
        self._prompt_builder = prompt_builder

    async def rewrite(self, text: str, contract: Any) -> str:
        """
        执行 Shadow Rewrite。

        Args:
            text: 原始文本
            contract: PlanningContract（dict 或对象）

        Returns:
            改写后的文本

        Raises:
            Exception: LLM 调用异常（由调用方处理）
        """
        prompt = self._prompt_builder.build(text, contract)
        return await self._llm_client.generate(prompt)