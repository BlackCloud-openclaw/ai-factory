"""
提取阶段数据模型 - 深度不可变
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any, Mapping, Tuple
from enum import Enum

from ..corpus.failure_mode import FailureMode


class FailureSource(Enum):
    LOG = "log"
    RUNTIME = "runtime"
    REGRESSION = "regression"
    MANUAL = "manual"
    PRODUCTION = "production"


@dataclass(frozen=True)
class RawFailureRecord:
    """原始失败记录（未标准化）"""
    source: FailureSource
    payload: Mapping[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    id: Optional[str] = None


@dataclass(frozen=True)
class NormalizedFailure:
    """标准化后的失败记录 - 深度不可变"""
    id: str
    timestamp: datetime
    failure_type: str
    severity: str
    message: str

    # 深度不可变：使用 Tuple 而非 List
    scene_text: Optional[str] = None
    planning_contract: Optional[Mapping[str, Any]] = None
    events: Optional[Tuple[Mapping[str, Any], ...]] = None
    snapshot_before: Optional[Mapping[str, Any]] = None
    snapshot_after: Optional[Mapping[str, Any]] = None
    runtime_metrics: Optional[Mapping[str, Any]] = None

    draft_before: Optional[str] = None
    draft_after: Optional[str] = None

    chapter: Optional[int] = None
    scene_idx: Optional[int] = None
    source: FailureSource = FailureSource.LOG
    tags: Tuple[str, ...] = ()  # 深度不可变


@dataclass(frozen=True)
class ClassifiedFailure:
    """包含分类信息的失败记录"""
    normalized: NormalizedFailure
    failure_mode: FailureMode