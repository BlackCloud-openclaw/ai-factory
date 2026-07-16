# src/runtime/revision_engine.py
"""
Revision Engine - 执行 PatchPlan
"""

import httpx
import logging
from openai import AsyncOpenAI
from dataclasses import dataclass
from typing import Optional

from src.runtime.patch_compiler import PatchPlan, PatchOperation
from src.runtime.patch_renderer import PatchRenderer

logger = logging.getLogger(__name__)


@dataclass
class RevisionResult:
    original_text: str
    patched_text: str
    modified: bool
    actions_applied: int
    error: Optional[str] = None


class RevisionEngine:
    def __init__(self, llm_api_base: str, llm_model: str):
        self.llm_api_base = llm_api_base
        self.llm_model = llm_model
        # 创建自定义 HTTP 客户端，禁用环境代理
        self._http_client = httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(120.0, connect=10.0))
        self._openai_client = AsyncOpenAI(
            api_key="not-needed",
            base_url=llm_api_base,
            http_client=self._http_client,
        )

    async def revise(self, draft: str, plan: PatchPlan) -> RevisionResult:
        if not plan.revision_required:
            return RevisionResult(
                original_text=draft,
                patched_text=draft,
                modified=False,
                actions_applied=0,
            )

        renderer = PatchRenderer()
        patch_prompt = renderer.render(plan, draft)

        if not patch_prompt:
            return RevisionResult(
                original_text=draft,
                patched_text=draft,
                modified=False,
                actions_applied=0,
                error="无法生成修订 Prompt",
            )

        try:
            response = await self._openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": patch_prompt}],
                temperature=0.3,
                max_tokens=2048,
            )
            patched = response.choices[0].message.content or draft
            return RevisionResult(
                original_text=draft,
                patched_text=patched,
                modified=True,
                actions_applied=len(plan.actions),
            )
        except Exception as e:
            logger.error(f"Revision failed: {e}")
            return RevisionResult(
                original_text=draft,
                patched_text=draft,
                modified=False,
                actions_applied=0,
                error=str(e),
            )
