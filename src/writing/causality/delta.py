"""Predicate Delta 结构 - 表示投影变化"""
from dataclasses import dataclass, field
from typing import List, Tuple
from .predicate import Predicate


@dataclass(frozen=True)
class PredicateRef:
    """用于标识要失效的谓词（不需要完整对象）"""
    identity_key: str
    event_id: int  # 失效事件ID


@dataclass(frozen=True)
class PredicateDelta:
    """投影变化：激活一批谓词，失效另一批谓词"""
    novel_id: str
    event_id: int
    projection_version: int
    event_semantic: str
    to_activate: List[Predicate] = field(default_factory=list)
    to_deactivate: List[PredicateRef] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.to_activate and not self.to_deactivate