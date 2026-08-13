# src/narrative/intent/__init__.py

from .model import (
    IntentSource,
    IntentPriority,
    NarrativeIntent,
    NarrativeIntentSet,
)
from .dimension import (
    IntentDimension,
    IntentDirection,
    BuiltinDimensions,
)
from .conflict import ConflictType, Conflict, detect_direction_conflicts
from .resolver import (
    ResolutionPlan,
    ResolutionStrategy,
    IntentResolver,
    resolve_intents,
    ConflictResolutionError,
)
from .legacy import LegacyIntentLoader
from .satisfaction import (
    SatisfactionItem,
    SatisfactionReport,
    SatisfactionEvaluator,
    KeywordSatisfactionEvaluator,
    IntentSatisfaction,
    evaluate_satisfaction,
    EvaluationResult,
)
from .llm_evaluator import LLMSemanticEvaluator  # 🆕 添加这一行
from ..conflict import ConflictResolution, ConflictStrategy

__all__ = [
    "IntentSource",
    "IntentPriority",
    "NarrativeIntent",
    "NarrativeIntentSet",
    "IntentDimension",
    "IntentDirection",
    "BuiltinDimensions",
    "ConflictType",
    "Conflict",
    "detect_direction_conflicts",
    "ResolutionPlan",
    "ResolutionStrategy",
    "IntentResolver",
    "resolve_intents",
    "ConflictResolutionError",
    "LegacyIntentLoader",
    "SatisfactionItem",
    "SatisfactionReport",
    "SatisfactionEvaluator",
    "KeywordSatisfactionEvaluator",
    "IntentSatisfaction",
    "evaluate_satisfaction",
    "EvaluationResult",
    "LLMSemanticEvaluator",      # 🆕 添加到导出列表
    "ConflictResolution",
    "ConflictStrategy",
]