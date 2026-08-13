# src/writing/audit/field_comparator.py
"""
Phase 10.2: FieldComparator — 比较字段值的变化
"""

from dataclasses import dataclass
from typing import Any, List, Dict, Set
from enum import Enum, auto


class Existence(Enum):
    """字段是否存在。"""
    PRESENT = "present"
    REMOVED = "removed"
    UNKNOWN = "unknown"


class ChangeType(Enum):
    """字段是否发生变化。"""
    UNCHANGED = "unchanged"
    MODIFIED = "modified"    # 值改变（非结构）
    PARTIAL = "partial"      # 部分保留（结构变化，如列表减少）


@dataclass(frozen=True)
class ComparisonResult:
    """
    字段比较结果，区分存在性和变化性。
    """
    existence: Existence
    change: ChangeType
    retention_ratio: float = 1.0   # 0.0 ~ 1.0，用于量化保留比例
    old_value: Any = None
    new_value: Any = None

    @property
    def is_present(self) -> bool:
        return self.existence == Existence.PRESENT

    @property
    def is_removed(self) -> bool:
        return self.existence == Existence.REMOVED

    @property
    def is_unchanged(self) -> bool:
        return self.change == ChangeType.UNCHANGED


class FieldComparator:
    """
    比较两个字段值，返回 ComparisonResult。
    """

    @staticmethod
    def compare(old_value: Any, new_value: Any) -> ComparisonResult:
        # 处理 None
        if old_value is None and new_value is None:
            return ComparisonResult(
                existence=Existence.UNKNOWN,
                change=ChangeType.UNCHANGED,
                retention_ratio=0.0,
                old_value=None,
                new_value=None,
            )
        if old_value is None:
            return ComparisonResult(
                existence=Existence.UNKNOWN,
                change=ChangeType.UNCHANGED,
                retention_ratio=0.0,
                old_value=None,
                new_value=new_value,
            )
        if new_value is None:
            return ComparisonResult(
                existence=Existence.REMOVED,
                change=ChangeType.UNCHANGED,
                retention_ratio=0.0,
                old_value=old_value,
                new_value=None,
            )

        # 列表比较
        if isinstance(old_value, list) and isinstance(new_value, list):
            return FieldComparator._compare_lists(old_value, new_value)

        # 集合比较
        if isinstance(old_value, (set, frozenset)) and isinstance(new_value, (set, frozenset)):
            return FieldComparator._compare_sets(old_value, new_value)

        # 字典比较
        if isinstance(old_value, dict) and isinstance(new_value, dict):
            return FieldComparator._compare_dicts(old_value, new_value)

        # 基本类型
        if old_value == new_value:
            return ComparisonResult(
                existence=Existence.PRESENT,
                change=ChangeType.UNCHANGED,
                retention_ratio=1.0,
                old_value=old_value,
                new_value=new_value,
            )
        return ComparisonResult(
            existence=Existence.PRESENT,
            change=ChangeType.MODIFIED,
            retention_ratio=0.0,
            old_value=old_value,
            new_value=new_value,
        )

    @staticmethod
    def _compare_lists(old_list: List, new_list: List) -> ComparisonResult:
        if not old_list:
            return ComparisonResult(
                existence=Existence.PRESENT if new_list else Existence.REMOVED,
                change=ChangeType.UNCHANGED if not new_list else ChangeType.MODIFIED,
                retention_ratio=0.0,
                old_value=old_list,
                new_value=new_list,
            )
        if not new_list:
            return ComparisonResult(
                existence=Existence.REMOVED,
                change=ChangeType.UNCHANGED,
                retention_ratio=0.0,
                old_value=old_list,
                new_value=new_list,
            )
        # 计算保留比例（交集大小 / 旧列表大小）
        try:
            old_set = set(old_list)
            new_set = set(new_list)
        except TypeError:
            # 列表元素不可哈希（如 dict），回退到基于字符串的匹配
            old_strs = [str(x) for x in old_list]
            new_strs = [str(x) for x in new_list]
            old_set = set(old_strs)
            new_set = set(new_strs)
        retained = len(old_set & new_set)
        ratio = retained / len(old_set) if old_set else 0.0
        existence = Existence.PRESENT
        change = ChangeType.UNCHANGED if ratio == 1.0 else ChangeType.PARTIAL if ratio > 0 else ChangeType.MODIFIED
        return ComparisonResult(
            existence=existence,
            change=change,
            retention_ratio=ratio,
            old_value=old_list,
            new_value=new_list,
        )

    @staticmethod
    def _compare_sets(old_set: Set, new_set: Set) -> ComparisonResult:
        if not old_set:
            return ComparisonResult(
                existence=Existence.PRESENT if new_set else Existence.REMOVED,
                change=ChangeType.UNCHANGED if not new_set else ChangeType.MODIFIED,
                retention_ratio=0.0,
                old_value=old_set,
                new_value=new_set,
            )
        if not new_set:
            return ComparisonResult(
                existence=Existence.REMOVED,
                change=ChangeType.UNCHANGED,
                retention_ratio=0.0,
                old_value=old_set,
                new_value=new_set,
            )
        retained = len(old_set & new_set)
        ratio = retained / len(old_set)
        existence = Existence.PRESENT
        change = ChangeType.UNCHANGED if ratio == 1.0 else ChangeType.PARTIAL if ratio > 0 else ChangeType.MODIFIED
        return ComparisonResult(
            existence=existence,
            change=change,
            retention_ratio=ratio,
            old_value=old_set,
            new_value=new_set,
        )

    @staticmethod
    def _compare_dicts(old_dict: Dict, new_dict: Dict) -> ComparisonResult:
        if not old_dict:
            return ComparisonResult(
                existence=Existence.PRESENT if new_dict else Existence.REMOVED,
                change=ChangeType.UNCHANGED if not new_dict else ChangeType.MODIFIED,
                retention_ratio=0.0,
                old_value=old_dict,
                new_value=new_dict,
            )
        if not new_dict:
            return ComparisonResult(
                existence=Existence.REMOVED,
                change=ChangeType.UNCHANGED,
                retention_ratio=0.0,
                old_value=old_dict,
                new_value=new_dict,
            )
        retained_keys = set(old_dict.keys()) & set(new_dict.keys())
        ratio = len(retained_keys) / len(old_dict) if old_dict else 0.0
        existence = Existence.PRESENT
        change = ChangeType.UNCHANGED if ratio == 1.0 else ChangeType.PARTIAL if ratio > 0 else ChangeType.MODIFIED
        return ComparisonResult(
            existence=existence,
            change=change,
            retention_ratio=ratio,
            old_value=old_dict,
            new_value=new_dict,
        )