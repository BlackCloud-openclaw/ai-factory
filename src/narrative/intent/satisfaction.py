# src/narrative/intent/satisfaction.py

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable, Tuple, List, Mapping, Any, Optional
from types import MappingProxyType

from src.narrative.artifact import NarrativeArtifact
from src.narrative.intent.model import NarrativeIntent, NarrativeIntentSet


# ============================================================
# 新增：EvaluationResult（数据契约）
# ============================================================

@dataclass(frozen=True)
class EvaluationResult:
    """满意度评估结果（含诊断信息）"""
    score: float                     # 0.0 ~ 1.0
    reason: str = ""                 # 人类可读理由
    evidence: Tuple[str, ...] = ()   # 支撑证据
    evaluator: str = ""              # 评估器标识（如 "LLM", "Keyword"）
    fallback: bool = False           # 是否为降级结果

    def __post_init__(self):
        # 确保 score 在合法范围
        if not (0.0 <= self.score <= 1.0):
            object.__setattr__(self, "score", max(0.0, min(1.0, self.score)))


# ============================================================
# 协议更新
# ============================================================

@runtime_checkable
class SatisfactionEvaluator(Protocol):
    """评估器协议：返回 EvaluationResult，而非 float"""

    async def evaluate(
        self,
        artifact: NarrativeArtifact,
        intent: NarrativeIntent,
    ) -> EvaluationResult:
        ...


# ============================================================
# 关键词评估器（适配新协议）
# ============================================================

class KeywordSatisfactionEvaluator:
    """关键词匹配评估器（fallback）"""

    async def evaluate(
        self,
        artifact: NarrativeArtifact,
        intent: NarrativeIntent,
    ) -> EvaluationResult:
        if not intent.desired_effect:
            return EvaluationResult(
                score=1.0,
                reason="意图为空，视为已满足",
                evaluator="Keyword",
            )

        keywords = self._extract_keywords(intent.desired_effect)
        if not keywords:
            return EvaluationResult(
                score=0.5,
                reason="无法提取有效关键词",
                evaluator="Keyword",
            )

        text = artifact.text.lower()
        matched = sum(1 for kw in keywords if kw in text)
        score = min(1.0, matched / len(keywords))

        return EvaluationResult(
            score=score,
            reason=f"命中 {matched}/{len(keywords)} 个关键词",
            evidence=tuple(kw for kw in keywords if kw in text),
            evaluator="Keyword",
            fallback=False,
        )

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        import re
        # 提取长度 ≥2 的中文字符
        tokens = re.findall(r'[\u4e00-\u9fa5]{2,}', text)
        stopwords = {"的", "了", "是", "在", "和", "与", "或", "但", "而", "被", "把"}
        return [t for t in tokens if t not in stopwords]


# ============================================================
# IntentSatisfaction 适配新协议
# ============================================================

@dataclass
class SatisfactionItem:
    intent_id: str
    satisfied: bool
    score: float
    evidence: Tuple[str, ...] = field(default_factory=tuple)
    reason: str = ""


@dataclass(frozen=True)
class SatisfactionReport:
    overall: float
    items: Tuple[SatisfactionItem, ...] = field(default_factory=tuple)
    passed: bool = False
    metadata: Mapping[str, Any] = field(default_factory=MappingProxyType)

    def __post_init__(self):
        if not isinstance(self.metadata, MappingProxyType):
            object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict:
        return {
            "overall": self.overall,
            "passed": self.passed,
            "items": [
                {
                    "intent_id": i.intent_id,
                    "satisfied": i.satisfied,
                    "score": i.score,
                    "evidence": list(i.evidence),
                    "reason": i.reason,
                }
                for i in self.items
            ],
            "metadata": dict(self.metadata),
        }


class IntentSatisfaction:
    def __init__(
        self,
        evaluator: Optional[SatisfactionEvaluator] = None,
        threshold: float = 0.6,
    ):
        self._evaluator = evaluator or KeywordSatisfactionEvaluator()
        self._threshold = threshold

    async def evaluate(
        self,
        artifact: NarrativeArtifact,
        intents: NarrativeIntentSet,
    ) -> SatisfactionReport:
        if not intents:
            return SatisfactionReport(
                overall=1.0,
                items=(),
                passed=True,
                metadata=MappingProxyType({"reason": "no_intents"}),
            )

        items = []
        total_score = 0.0

        for intent in intents.intents:
            result = await self._evaluator.evaluate(artifact, intent)

            # 使用结果中的 score
            score = result.score
            total_score += score
            satisfied = score >= self._threshold

            items.append(
                SatisfactionItem(
                    intent_id=str(intent.id),
                    satisfied=satisfied,
                    score=score,
                    evidence=result.evidence,
                    reason=result.reason or f"达成度 {score:.0%}",
                )
            )

        overall = total_score / len(intents.intents) if intents.intents else 1.0
        passed = overall >= self._threshold

        return SatisfactionReport(
            overall=overall,
            items=tuple(items),
            passed=passed,
            metadata=MappingProxyType({
                "threshold": self._threshold,
                "evaluator_type": self._evaluator.__class__.__name__,
            }),
        )


async def evaluate_satisfaction(
    artifact: NarrativeArtifact,
    intents: NarrativeIntentSet,
    evaluator: Optional[SatisfactionEvaluator] = None,
    threshold: float = 0.6,
) -> SatisfactionReport:
    return await IntentSatisfaction(evaluator, threshold).evaluate(artifact, intents)