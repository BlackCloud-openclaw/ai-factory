# src/writing/audit/field_extractor.py
"""
Phase 10.2: FieldExtractor — 从 Payload 中提取字段值
"""

from dataclasses import dataclass
from typing import Any, Optional, List, Union


@dataclass(frozen=True)
class ExtractionResult:
    """提取结果，区分字段缺失与值为 None。"""
    found: bool
    value: Any = None


class FieldExtractor:
    """
    从 payload 中提取指定字段的值。

    支持：
        - 字典键查找（递归）
        - 列表元素（返回所有匹配的列表项）
    """

    def extract(self, data: Any, field_name: str) -> ExtractionResult:
        """
        提取字段值。

        Returns:
            ExtractionResult: found=True 表示字段存在，value 为对应的值（可能为 None）。
        """
        return self._extract_recursive(data, field_name)

    def _extract_recursive(self, data: Any, field_name: str) -> ExtractionResult:
        if isinstance(data, dict):
            if field_name in data:
                return ExtractionResult(found=True, value=data[field_name])
            for value in data.values():
                result = self._extract_recursive(value, field_name)
                if result.found:
                    return result
        elif isinstance(data, (list, tuple, set)):
            results = []
            for item in data:
                result = self._extract_recursive(item, field_name)
                if result.found:
                    results.append(result.value)
            if results:
                return ExtractionResult(found=True, value=results)
        return ExtractionResult(found=False, value=None)