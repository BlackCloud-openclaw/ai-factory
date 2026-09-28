"""
Phase 15.3 — Shadow Rewrite Layer

纯实验层，不参与生产 Runtime。
职责：在影子模式下执行 Rewrite，复用生产 Validator 结果。
"""

from .result import ShadowRewriteResult, ShadowRewriteStatus
from .runner import ShadowRewriteRunner, Rewriter, Validator
from .recorder import ShadowRecorder, MemoryShadowRecorder, DatabaseShadowRecorder
from .llm_client import LLMClient
from .rewriter import ShadowRewriter
from .validator import ShadowValidator, ShadowValidationResult
from .prompt_builder import ShadowPromptBuilder

__all__ = [
    "ShadowRewriteResult",
    "ShadowRewriteStatus",
    "ShadowRewriteRunner",
    "Rewriter",
    "Validator",
    "ShadowRecorder",
    "MemoryShadowRecorder",
    "DatabaseShadowRecorder",
    "LLMClient",
    "ShadowRewriter",
    "ShadowValidator",
    "ShadowValidationResult",
    "ShadowPromptBuilder",
]