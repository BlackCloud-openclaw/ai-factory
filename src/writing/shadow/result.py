# src/writing/shadow/result.py

from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List
from datetime import datetime
from enum import Enum


class ShadowRewriteStatus(str, Enum):
    """Rewrite 执行的最终状态。"""
    SUCCESS = "success"
    VALIDATION_FAILED = "validation_failed"
    LLM_ERROR = "llm_error"
    VALIDATOR_ERROR = "validator_error"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class ShadowRewriteResult:
    """
    单个场景的 Shadow Rewrite 完整记录。

    设计原则：
    - 不可变
    - 包含 Original 和 Rewrite 的完整对比信息
    - Original ValidationResult 由调用方传入，Runner 不重复验证
    """
    scene_id: str

    # Original 信息（由生产阶段传入）
    original_text: str
    original_validation_passed: bool
    original_violations: Optional[Any] = None

    # Rewrite 信息
    rewritten_text: Optional[str] = None
    rewritten_validation_passed: Optional[bool] = None
    rewritten_violations: Optional[Any] = None

    # 执行元数据
    status: ShadowRewriteStatus = ShadowRewriteStatus.SKIPPED
    error_message: Optional[str] = None
    experiment_id: Optional[str] = None
    prompt_version: str = "phase15.3.v1"
    contract_id: Optional[str] = None
    model: Optional[str] = None
    executed_at: datetime = field(default_factory=datetime.now)

    # A-1: contract_data（完整的 PlanningContract，用于离线重放）
    contract_data: Optional[Dict[str, Any]] = None

    # ========== Phase 15.5: 新增 writer_events ==========
    # None  = 数据不可用（历史记录或未采集）
    # []    = Writer 明确返回了空 events
    # [...] = 正常可诊断数据
    writer_events: Optional[List[Dict[str, Any]]] = None
    # ===================================================

    # 派生字段（__post_init__ 计算）
    original_length: int = field(init=False)
    rewritten_length: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'original_length', len(self.original_text))
        object.__setattr__(self, 'rewritten_length', len(self.rewritten_text) if self.rewritten_text else 0)

    @property
    def original_passed(self) -> bool:
        return self.original_validation_passed

    @property
    def rewritten_passed(self) -> Optional[bool]:
        """返回 Rewrite 的 Validator 结果。None 表示没有 Validator 结果（如 LLM_ERROR/TIMEOUT）。"""
        return self.rewritten_validation_passed

    @property
    def is_complete(self) -> bool:
        """Rewrite 是否成功执行（不论 Validator 结果）。"""
        return self.status in (ShadowRewriteStatus.SUCCESS, ShadowRewriteStatus.VALIDATION_FAILED)

    @property
    def is_regression(self) -> bool:
        """
        Rewrite 是否导致 Validator 结果退化。
        仅当 Original PASS 且 Rewrite 明确 FAIL 时为 True。
        LLM_ERROR / TIMEOUT / VALIDATOR_ERROR 不触发 regression。
        """
        return (
            self.original_validation_passed is True
            and self.rewritten_validation_passed is False
        )