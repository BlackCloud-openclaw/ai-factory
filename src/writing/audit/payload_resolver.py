# src/writing/audit/payload_resolver.py
"""
Phase 10.2: PayloadResolver — 从 PayloadRef 加载真实数据
"""

from typing import Any, Protocol, Dict, Iterable
from .payload_ref import PayloadRef


class PayloadResolver(Protocol):
    """
    根据 PayloadRef 加载 Payload 的协议。
    ExecutionTrace 只存储引用，Analyzer 通过此协议获取实际数据。
    """

    def resolve(self, ref: PayloadRef) -> Any:
        """
        根据引用加载数据。

        Args:
            ref: PayloadRef 对象

        Returns:
            解析后的数据（dict, list, str 等）

        Raises:
            ValueError: 如果引用无法解析
        """
        ...

    def resolve_many(self, refs: Iterable[PayloadRef]) -> Dict[str, Any]:
        """
        批量解析引用，返回 {str(ref): data} 字典。
        无法解析的引用对应值为 None。
        """
        ...


class MemoryPayloadResolver:
    """
    内存中的 PayloadResolver（用于测试和开发）。
    """

    def __init__(self):
        self._storage: Dict[str, Any] = {}

    def register(self, ref: PayloadRef, data: Any) -> None:
        self._storage[str(ref)] = data

    def resolve(self, ref: PayloadRef) -> Any:
        key = str(ref)
        if key not in self._storage:
            raise ValueError(f"PayloadRef not found: {key}")
        return self._storage[key]

    def resolve_many(self, refs: Iterable[PayloadRef]) -> Dict[str, Any]:
        result = {}
        for ref in refs:
            try:
                result[str(ref)] = self.resolve(ref)
            except ValueError:
                result[str(ref)] = None
        return result