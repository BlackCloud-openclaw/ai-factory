"""
SemanticJudge - 使用 LLM 判断证据是否支持 Claim

关键设计：
- 惰性创建 pool，避免在无 event loop 时构造
- 直接使用 httpx.AsyncClient，客户端自动关闭

C3.4.1 修订:
- plot_flag prompt 明确三值语义边界：
    SUPPORTED      事实已发生
    CONTRADICTED   事实未发生 / 与命题冲突
    INSUFFICIENT   场景未提供足够信息
  目的是让"无信息"与"反证"两类案例在 Prompt 层即可区分，
  与 Validator 的三值 verdict 严格对齐。
"""

import json
import re
import logging
from typing import List, Optional, Dict, Any
import httpx

from .models import VerificationClaim, Evidence, SemanticJudgement, Verdict

logger = logging.getLogger(__name__)


class SemanticJudge:
    """语义法官 - 使用 LLM 判断证据是否支持 Claim"""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 8192,
        timeout: float = 1200.0,
    ):
        self.model = model or "Qwen3-32B-Q5_K_M"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._pool = None  # 惰性创建，避免无 event loop 时构造

    @property
    def pool(self):
        """惰性获取 pool"""
        if self._pool is None:
            from src.execution.llm_router_pool import get_llm_router_pool
            self._pool = get_llm_router_pool()
        return self._pool

    async def judge(
        self,
        claim: VerificationClaim,
        evidence_list: List[Evidence],
    ) -> SemanticJudgement:
        """判断证据是否支持 Claim"""
        if not evidence_list:
            return SemanticJudgement(
                verdict=Verdict.INSUFFICIENT,
                confidence=0.0,
                reason="No evidence provided",
                evidence_ids=[],
            )

        prompt = self._build_prompt(claim, evidence_list)

        try:
            response_text = await self._call_llm(prompt)
            result = self._parse_response(response_text)

            verdict_map = {
                "SUPPORTED": Verdict.SUPPORTED,
                "CONTRADICTED": Verdict.CONTRADICTED,
                "INSUFFICIENT": Verdict.INSUFFICIENT,
            }

            verdict = verdict_map.get(
                result.get("verdict", "INSUFFICIENT"),
                Verdict.INSUFFICIENT,
            )
            confidence = float(result.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            return SemanticJudgement(
                verdict=verdict,
                confidence=confidence,
                reason=result.get("reason", "LLM judgement"),
                evidence_ids=[e.evidence_id for e in evidence_list[:3]],
            )

        except Exception as e:
            logger.error(f"[SemanticJudge] LLM call failed: {e}")
            return SemanticJudgement(
                verdict=Verdict.INSUFFICIENT,
                confidence=0.0,
                reason=f"LLM error: {e}",
                evidence_ids=[],
            )

    async def _call_llm(self, prompt: str) -> str:
        """直接使用 httpx.AsyncClient 调用 LLM"""
        timeout = httpx.Timeout(self.timeout, connect=10.0)
        base_url = self.pool.get_base_url(self.model)

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "chat_template_kwargs": {"enable_thinking": False},  # ← 加这一行
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"] or ""

    def _build_prompt(
        self,
        claim: VerificationClaim,
        evidence_list: List[Evidence],
    ) -> str:
        evidence_text = "\n".join(
            [f"- {e.text[:300]}" for e in evidence_list[:5]]
        )

        type_specific = ""
        if claim.state_change_type == "realm_change":
            type_specific = """
## realm_change 语义条件

判断标准是：**证据是否确认"境界转换"这一事实已经完成**。

- 特征描述（"展现大乘威压"、"气息接近大乘"）→ 不构成事实完成
- 意图/尝试（"试图突破"、"正在冲击大乘"）→ 不构成事实完成
- 已完成转换（无论用什么自然语言表达）→ 构成事实完成
- 明确否定（"突破失败"）→ CONTRADICTED

注意：**不要求证据中出现任何特定词语**。判断依据是语义等价性。
"""
        elif claim.state_change_type == "knowledge_gain":
            type_specific = """
## knowledge_gain 语义条件

- 完成的获得/理解（"他理解了..."、"他领悟了..."）→ SUPPORTED
- 尝试/未完成（"他试图理解..."）→ INSUFFICIENT
- 仅听闻未内化（"他听说过..."）→ INSUFFICIENT
"""
        elif claim.state_change_type == "plot_flag":
            # ============================================================
            # C3.4.1 修订：明确三值语义边界
            # ============================================================
            type_specific = """
## plot_flag 语义条件

判断标准是：**证据是否足以确认该事实已经发生**。

三值语义边界（严格区分）:

1. SUPPORTED — 证据表明事实已经发生
   - 结果性证据（事件产生的直接后果，如"封印亮起"、"符文流转"）
   - 明确的完成陈述（如"封印被触发了"、"他终于触发了封印"）

2. CONTRADICTED — 证据明确表明事实**没有发生**，与命题冲突
   - 明确描述事实处于未触发/未成立状态（如"封印沉睡着"、"毫无动静"）
   - 明确否定的陈述（如"封印并未触发"、"什么都没有发生"）

3. INSUFFICIENT — 场景未提供足够信息判断事实是否发生
   - 完全未提及该事实
   - 仅提及相关事物但未描述其状态（如"他站在石壁前"）

关键区别:
- "沉睡、毫无动静" → CONTRADICTED（明确否定）
- "未提及封印状态" → INSUFFICIENT（无信息）

不要求证据中直接出现 flag 名称。语义等价的表达即可。
"""
            # ============================================================

        return f"""你是一位验证法官。判断证据是否足以确认命题所述的事实已经成立。

## 命题
{claim.proposition}

## 证据
{evidence_text if evidence_text else "（无证据）"}

## 通用判断规则

- SUPPORTED: 证据明确证明命题的事实**已经完成/成立**
- CONTRADICTED: 证据明确否定命题
- INSUFFICIENT: 证据不足或仅表示倾向/特征/尝试

核心原则：
- **特征、意图、未来可能性不构成事实完成**
- 判断依据是语义等价性，不要求特定词语出现
{type_specific}

## 输出格式 (JSON only)
{{
    "verdict": "SUPPORTED" | "CONTRADICTED" | "INSUFFICIENT",
    "confidence": 0.0-1.0,
    "reason": "判断理由（简短）"
}}

只输出 JSON。"""

    def _parse_response(self, response: str) -> Dict[str, Any]:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if not match:
            return {
                "verdict": "INSUFFICIENT",
                "confidence": 0.0,
                "reason": "Parse error",
            }
        try:
            result = json.loads(match.group())
            if "reason" in result and isinstance(result["reason"], str):
                result["reason"] = re.sub(
                    r'[\x00-\x1f\x7f-\x9f]', '', result["reason"]
                )
            return result
        except json.JSONDecodeError:
            return {
                "verdict": "INSUFFICIENT",
                "confidence": 0.0,
                "reason": "Invalid JSON",
            }