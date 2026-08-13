# src/writing/validation/__init__.py
"""
Phase 13.2.3B + 13.2.3C: Semantic Validator & Quality Gate
Public API exports.
"""

from .evidence import ValidationEvidence, ValidationResult
from .signal_weight import SignalWeightPolicy
from .matchers import (
    Matcher,
    ExactMatcher,
    NormalizedMatcher,
    KeywordCoverageMatcher,
    MatcherResult,
)
from .embedding_provider import EmbeddingProvider
from .embedding_matcher import (
    EmbeddingMatcher,
    NoOpEmbeddingProvider,
)
from .semantic_validator import SemanticValidator

# 导出 SignalSource（从 planning_contract 导入）
from ..planning_contract import SignalSource

# 不再从 quality_gate 导入，避免循环导入
# QualityGate 应该直接从 quality_gate 模块导入

__all__ = [
    "ValidationEvidence",
    "ValidationResult",
    "SignalWeightPolicy",
    "Matcher",
    "ExactMatcher",
    "NormalizedMatcher",
    "KeywordCoverageMatcher",
    "MatcherResult",
    "EmbeddingProvider",
    "EmbeddingMatcher",
    "NoOpEmbeddingProvider",
    "SemanticValidator",
    "SignalSource",  # 新增
]