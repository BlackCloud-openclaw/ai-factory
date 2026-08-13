# src/writing/validation/evidence.py
"""
Validation Evidence & Result Models

Phase 13.2.3B + 13.2.3C: 
- B-4, B-13, B-14
- C: blocking_missing 字段
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import hashlib

from src.writing.planning_contract import SignalSource
from .models import MissingContractChange

@dataclass
class ValidationEvidence:
    """单个验证事件的证据 (B-4, B-13)。"""
    evidence_id: str
    event_id: str
    event_text: str
    matcher: str
    confidence: float
    source: SignalSource
    matched_text: str
    weight: float = 1.0
    details: Optional[Dict[str, Any]] = None

    @classmethod
    def generate_id(cls, scene_id: str, event_id: str, matcher: str, matched_text: str) -> str:
        """生成稳定 evidence_id (B-13)。"""
        raw = f"{scene_id}|{event_id}|{matcher}|{matched_text[:50]}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


@dataclass
class ValidationResult:
    """
    完整验证结果 (B-4, B-14)。

    Phase 13.2.3C: 增加 blocking_missing 字段，
    使 QualityGate 可以强类型消费阻断性缺失，无需解析字符串。
    """
    passed: bool
    missing: List[str]                          # 所有未匹配的事件
    matched: List[ValidationEvidence]           # 匹配的证据
    blocking_missing: List[str] = field(default_factory=list)  # Phase 13.2.3C 新增
    overall_confidence: float = 0.0
    weight_applied: float = 0.0
    validation_version: str = "13.2.3B-v1.1"
    errors: List[str] = field(default_factory=list)
    
    # 新增：缺失的契约投影列表
    missing_changes: List[MissingContractChange] = field(default_factory=list)

    @property
    def match_count(self) -> int:
        return len(self.matched)

    @property
    def missing_count(self) -> int:
        return len(self.missing)

    @property
    def blocking_missing_count(self) -> int:
        return len(self.blocking_missing)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典 (用于审计/日志)。"""
        return {
            "passed": self.passed,
            "missing_count": self.missing_count,
            "match_count": self.match_count,
            "blocking_missing_count": self.blocking_missing_count,
            "overall_confidence": round(self.overall_confidence, 3),
            "weight_applied": round(self.weight_applied, 3),
            "validation_version": self.validation_version,
            "missing_events": self.missing[:5],
            "blocking_missing": self.blocking_missing[:5],
            "matched": [
                {
                    "event_id": e.event_id,
                    "matcher": e.matcher,
                    "confidence": round(e.confidence, 3),
                    "source": e.source.value if hasattr(e.source, 'value') else str(e.source),
                    "weight": round(e.weight, 3),
                }
                for e in self.matched[:5]
            ],
            "errors": self.errors[:3],
        }