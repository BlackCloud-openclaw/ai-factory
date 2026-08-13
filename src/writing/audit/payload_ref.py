# src/writing/audit/payload_ref.py
"""
Phase 10.2: PayloadRef — 运行时数据引用（URI 字符串）
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PayloadRef:
    """
    运行时数据引用（轻量级）。

    Attributes:
        uri: 统一资源标识符，例如 "snapshot://abc123", "memory://planning/001"
    """
    uri: str

    def __str__(self) -> str:
        return self.uri

    @property
    def scheme(self) -> str:
        """解析 scheme（如 "snapshot"）。"""
        if "://" in self.uri:
            return self.uri.split("://", 1)[0]
        return ""

    @property
    def identifier(self) -> str:
        """解析 identifier（scheme 后的部分）。"""
        if "://" in self.uri:
            return self.uri.split("://", 1)[1]
        return self.uri