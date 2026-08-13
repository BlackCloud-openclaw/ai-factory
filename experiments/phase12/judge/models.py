"""
LLM Judge 数据模型
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Dict
from enum import Enum


class JudgeDimension(Enum):
    """LLM Judge 评估维度"""
    CONTINUITY = "continuity"
    CHARACTER = "character"
    DIALOGUE = "dialogue"
    FLOW = "flow"


@dataclass(frozen=True)
class JudgeResult:
    """LLM Judge 单次评估结果"""
    dimension: JudgeDimension
    score: float  # 0.0 - 1.0
    confidence: float  # 0.0 - 1.0
    reasoning: str
    tokens_used: int
    elapsed_ms: int
    raw_response: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class JudgeCacheKey:
    """LLM Judge 缓存键"""
    dimension: str
    text_hash: str
    prompt_version: str
    model: str


@dataclass(frozen=True)
class JudgeCacheEntry:
    """LLM Judge 缓存条目"""
    result: JudgeResult
    timestamp: float
    ttl: int = 86400  # 24 小时