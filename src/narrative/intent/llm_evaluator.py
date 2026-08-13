# src/narrative/intent/llm_evaluator.py
"""
LLM 驱动的语义满意度评估器（Phase 9.2.2）
"""

import json
import logging
from typing import Optional

from src.narrative.artifact import NarrativeArtifact
from src.narrative.intent.model import NarrativeIntent
from src.narrative.intent.satisfaction import (
    SatisfactionEvaluator,
    EvaluationResult,
)
from src.narrative.realizers.interfaces import TextGenerator

logger = logging.getLogger(__name__)


class LLMSemanticEvaluator:
    """
    语义评估器，使用 LLM 判断意图是否实现。
    遵循 SatisfactionEvaluator 协议，但无需显式继承。
    """

    def __init__(
        self,
        text_generator: TextGenerator,
        prompt_template: Optional[str] = None,
        fallback_evaluator: Optional[SatisfactionEvaluator] = None,
    ):
        self._generator = text_generator
        self._fallback_evaluator = fallback_evaluator or self._default_fallback()
        self._prompt_template = prompt_template or self._default_prompt()

    def _default_prompt(self) -> str:
        return """你是一位资深叙事分析师，擅长评估文本是否实现了特定的叙事意图。

## 意图描述
{desired_effect}

## 意图维度
维度: {dimension}
方向: {direction}

## 待评估的文本
{text}

## 任务
请判断上述文本是否实现了“意图描述”中的目标，同时考虑维度和方向。
考虑因素：
- 意图的核心语义是否在文本中有所体现？
- 如果是隐含的，是否足够明显让读者感知到？
- 方向是“增加/减少/转变/稳定”，文本是否符合？

## 评分指南
请先分析，然后给出完成度分数（0-100），其中：
- 0-20：完全未实现
- 21-40：仅微弱体现
- 41-60：部分实现，有明显差距
- 61-80：较好地实现，仍有提升空间
- 81-100：完美实现

## 输出要求
请严格输出 JSON 格式：
{{
    "score": 85,          // 0-100 整数
    "reason": "简短理由",
    "evidence": ["证据1", "证据2"]
}}
只输出 JSON，不要有任何额外文字。"""

    def _default_fallback(self) -> SatisfactionEvaluator:
        from src.narrative.intent.satisfaction import KeywordSatisfactionEvaluator
        return KeywordSatisfactionEvaluator()

    async def evaluate(
        self,
        artifact: NarrativeArtifact,
        intent: NarrativeIntent,
    ) -> EvaluationResult:
        """返回 EvaluationResult，包含分数、理由、证据。"""
        if not intent.desired_effect:
            return EvaluationResult(
                score=1.0,
                reason="意图为空，视为已满足",
                evaluator="LLM",
                fallback=False,
            )

        text_sample = artifact.text[:4000]

        prompt = self._prompt_template.format(
            desired_effect=intent.desired_effect,
            dimension=intent.dimension.id,
            direction=intent.dimension.direction.value,
            text=text_sample,
        )

        try:
            response = await self._generator.generate(prompt)
            cleaned = self._clean_response(response)
            data = json.loads(cleaned)

            raw_score = data.get("score", 0)
            if isinstance(raw_score, (int, float)):
                if raw_score > 1:
                    score = raw_score / 100.0
                else:
                    score = raw_score
            else:
                score = 0.5

            score = max(0.0, min(1.0, score))

            reason = data.get("reason", "LLM 未提供理由")
            evidence = tuple(data.get("evidence", []))

            # ✅ 修复：intent.id 是 UUID，转为字符串再切片
            logger.debug(
                f"Semantic eval for intent {str(intent.id)[:8]}...: "
                f"score={score:.2f}, reason={reason[:50]}..."
            )

            return EvaluationResult(
                score=score,
                reason=reason,
                evidence=evidence,
                evaluator="LLM",
                fallback=False,
            )

        except json.JSONDecodeError as e:
            logger.warning(f"LLM response JSON parse failed: {e}. Falling back.")
            return await self._fallback_evaluator.evaluate(artifact, intent)

        except Exception as e:
            logger.error(f"LLM semantic evaluation failed: {e}", exc_info=True)
            return await self._fallback_evaluator.evaluate(artifact, intent)

    def _clean_response(self, response: str) -> str:
        """移除 markdown 代码块，提取 JSON"""
        response = response.strip()
        if "```json" in response:
            parts = response.split("```json")
            if len(parts) > 1:
                response = parts[1].split("```")[0].strip()
        elif "```" in response:
            parts = response.split("```")
            if len(parts) > 1:
                response = parts[1].split("```")[0].strip()
        return response