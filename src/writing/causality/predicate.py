"""Predicate 值对象 - 不可变，支持 identity 和规范化"""
import json
from dataclasses import dataclass
from typing import Any, Optional
from src.common.canonical import canonical_dumps


def normalize_object(obj: Any) -> str:
    if isinstance(obj, dict):
        return canonical_dumps(obj)    
    elif isinstance(obj, list):
        # 递归处理列表
        return canonical_dumps(obj)
    elif isinstance(obj, str):
        return obj.lower().strip()
    elif hasattr(obj, 'value'):  # Enum 或类似
        return str(obj.value).lower().strip()
    else:
        return str(obj).lower().strip()


@dataclass(frozen=True)
class Predicate:
    """不可变谓词值对象"""
    subject: str
    relation: str
    object: Any
    negated: bool = False
    confidence: float = 1.0
    priority: str = 'narrative'      # 'core', 'narrative', 'flavor'
    scope: str = 'scene'
    source_event_id: Optional[int] = None
    source_event_type: Optional[str] = None      # 新增
    source_event_semantic: Optional[str] = None

    def identity_key(self) -> str:
        """返回唯一标识该谓词的键（规范化 subject/relation/object）"""
        subj = self.subject.lower().strip()
        rel = self.relation.lower().strip()
        obj_str = normalize_object(self.object)
        neg = "not_" if self.negated else ""
        return f"{subj}|{rel}|{neg}{obj_str}"

    def __post_init__(self):
        # 验证置信度范围
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be in [0,1], got {self.confidence}")
        # 验证 priority 合法值
        if self.priority not in ('core', 'narrative', 'flavor'):
            raise ValueError(f"Priority must be core/narrative/flavor, got {self.priority}")