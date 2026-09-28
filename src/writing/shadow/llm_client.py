"""
Phase 15.3 — LLM Client

独立的 LLM 客户端，不依赖 ControlledWriter 或 WritingAgent。
仅负责 OpenAI-compatible LLM 调用。
"""

import logging
from typing import Optional
from openai import AsyncOpenAI
import httpx

logger = logging.getLogger(__name__)


class LLMClient:
    """
    独立的 LLM 客户端。

    职责：
    - 接收 prompt
    - 调用 OpenAI-compatible endpoint
    - 返回纯文本
    - 异常直接抛出，由调用方处理
    """

    def __init__(
        self,
        api_base: str,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        timeout: float = 120.0,
    ):
        self.api_base = api_base
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    async def generate(self, prompt: str) -> str:
        """
        生成文本。

        Raises:
            Exception: 任何 LLM 调用异常（由调用方处理）
        """
        transport = httpx.AsyncHTTPTransport(proxy=None)
        async with httpx.AsyncClient(
            transport=transport,
            timeout=httpx.Timeout(self.timeout, connect=30.0)
        ) as client:
            openai_client = AsyncOpenAI(
                api_key="not-needed",
                base_url=self.api_base,
                http_client=client,
            )
            response = await openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                # 注意：不设置 response_format，保持纯文本输出
            )
            content = response.choices[0].message.content or ""
            return content