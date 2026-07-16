# src/runtime/revision_trace.py
"""
Revision Trace - 修订执行可观测数据
"""

from dataclasses import dataclass, field
from typing import Optional
from src.runtime.validator import ComplianceReport
from src.runtime.patch_compiler import PatchPlan


@dataclass(frozen=True)
class RevisionTrace:
    """修订执行的完整追踪"""
    # 输入
    original_text: str
    patch_plan: PatchPlan
    patch_prompt: str
    
    # 执行
    revised_text: str
    diff: str                       # 文本差异（如 unified diff 或简单的字符变化统计）
    token_count_before: int
    token_count_after: int
    
    # 验证
    validator_before: ComplianceReport
    validator_after: ComplianceReport
    
    # 元数据
    revision_strategy: str
    target_layers: list[str]
    timestamp: float = field(default_factory=lambda: __import__('time').time())
    
    @property
    def has_change(self) -> bool:
        return self.original_text != self.revised_text
    
    @property
    def compliance_improved(self) -> bool:
        return self.validator_after.compliance_rate > self.validator_before.compliance_rate
    
    @property
    def compliance_degraded(self) -> bool:
        return self.validator_after.compliance_rate < self.validator_before.compliance_rate
    
    @property
    def diff_summary(self) -> str:
        """简短的差异摘要"""
        if not self.has_change:
            return "无变化"
        added = len(self.revised_text) - len(self.original_text)
        return f"文本长度变化: {added:+d} 字符"