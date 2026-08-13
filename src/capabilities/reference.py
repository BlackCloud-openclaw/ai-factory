# src/capabilities/reference.py

from dataclasses import dataclass
from typing import Optional
from packaging.version import Version


@dataclass(frozen=True)
class CapabilityRef:
    """
    Capability 引用

    - id: 全局唯一标识
    - version: 可选版本锁定（None 表示使用最新）
    """
    id: str
    version: Optional[Version] = None

    def __str__(self) -> str:
        if self.version:
            return f"{self.id}@{self.version}"
        return self.id

    @classmethod
    def parse(cls, s: str) -> "CapabilityRef":
        """从字符串解析，支持 "builtin.keyword@1.0.0" 格式"""
        if "@" in s:
            id_part, version_part = s.split("@", 1)
            if not id_part:
                raise ValueError(f"Invalid CapabilityRef: missing id in '{s}'")
            if not version_part:
                raise ValueError(f"Invalid CapabilityRef: missing version in '{s}'")
            try:
                version = Version(version_part)
            except Exception as e:
                raise ValueError(f"Invalid version in CapabilityRef '{s}': {e}")
            return cls(id=id_part, version=version)

        if not s:
            raise ValueError("Invalid CapabilityRef: empty string")
        return cls(id=s)