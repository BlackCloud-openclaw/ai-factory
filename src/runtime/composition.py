# src/runtime/composition.py
"""
Composition Root - Phase 9 依赖注入配置
所有基础设施依赖在此组装。
"""

from src.narrative.intent import (
    IntentSatisfaction,
    LLMSemanticEvaluator,
    KeywordSatisfactionEvaluator,
)
from src.narrative.realizers.interfaces import TextGenerator


def create_default_satisfaction(
    text_generator: TextGenerator,
    threshold: float = 0.6,
) -> IntentSatisfaction:
    """
    创建默认的满意度评估器。
    优先使用 LLMSemanticEvaluator，失败降级到 Keyword。
    """
    semantic = LLMSemanticEvaluator(
        text_generator=text_generator,
        fallback_evaluator=KeywordSatisfactionEvaluator(),
    )
    return IntentSatisfaction(
        evaluator=semantic,
        threshold=threshold,
    )


# 也可以直接暴露评估器，供其他模块使用
def create_llm_evaluator(
    text_generator: TextGenerator,
    fallback_to_keyword: bool = True,
) -> LLMSemanticEvaluator:
    fallback = KeywordSatisfactionEvaluator() if fallback_to_keyword else None
    return LLMSemanticEvaluator(
        text_generator=text_generator,
        fallback_evaluator=fallback,
    )