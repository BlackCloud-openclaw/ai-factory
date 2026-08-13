"""
Phase 12.2A: Evaluation Data Contract

定义 Writer 执行后的标准化评估产物。
这是 Benchmark、Corpus、Writer 三方共享的数据协议。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime
from uuid import uuid4, UUID
from dataclasses import asdict

@dataclass(frozen=True)
class RuntimeMetrics:
    """Writer 执行的运行时指标"""
    retry_count: int = 0
    fallback_count: int = 0
    error_count: int = 0
    validation_score: float = 1.0
    execution_time_ms: int = 0
    llm_calls: int = 0
    total_tokens: int = 0
    segments_total: int = 0
    segments_succeeded: int = 0
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RevisionResult:
    """修订结果（仅当实际执行了修订时存在）"""
    before_compliance: float
    after_compliance: float

    @property
    def delta(self) -> float:
        return self.after_compliance - self.before_compliance

    def to_dict(self) -> dict:
        return {
            "before_compliance": self.before_compliance,
            "after_compliance": self.after_compliance,
            "delta": self.delta,
        }

@dataclass(frozen=True)
class JudgeContext:
    """供 LLM Judge 使用的上下文"""
    previous_scene_text: Optional[str] = None
    character_summary: Optional[str] = None
    world_summary: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass(frozen=True)
class EvaluationSnapshot:
    """
    Writer 单次执行后的完整评估快照。

    设计原则：
    - 不可变（frozen=True）
    - 不包含 Runtime 内部状态
    - 所有字段都是 Writer 执行后的事实记录
    - artifacts 保留扩展性
    """

    # 输入/输出
    scene_before: str
    scene_after: str

    # 运行时指标（必须有值）
    runtime_metrics: RuntimeMetrics

    # 修订结果（可选：无修订时为 None）
    revision_result: Optional[RevisionResult] = None

    # Judge 上下文（由 ContextFactory 填充）
    judge_context: Optional[JudgeContext] = None

    # 扩展字段（用于未来新增评估维度）
    artifacts: Dict[str, Any] = field(default_factory=dict)

    # 元数据
    snapshot_id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "snapshot_id": str(self.snapshot_id),
            "created_at": self.created_at.isoformat(),
            "scene_before": self.scene_before,
            "scene_after": self.scene_after,
            "runtime_metrics": {
                "retry_count": self.runtime_metrics.retry_count,
                "fallback_count": self.runtime_metrics.fallback_count,
                "error_count": self.runtime_metrics.error_count,
                "validation_score": self.runtime_metrics.validation_score,
                "execution_time_ms": self.runtime_metrics.execution_time_ms,
                "llm_calls": self.runtime_metrics.llm_calls,
                "total_tokens": self.runtime_metrics.total_tokens,
                "segments_total": self.runtime_metrics.segments_total,
                "segments_succeeded": self.runtime_metrics.segments_succeeded,
            },
            "revision_result": {
                "before_compliance": self.revision_result.before_compliance,
                "after_compliance": self.revision_result.after_compliance,
                "delta": self.revision_result.delta,
            } if self.revision_result else None,
            "judge_context": {
                "previous_scene_text": self.judge_context.previous_scene_text if self.judge_context else None,
                "character_summary": self.judge_context.character_summary if self.judge_context else None,
                "world_summary": self.judge_context.world_summary if self.judge_context else None,
            } if self.judge_context else None,
            "artifacts": self.artifacts,
        }