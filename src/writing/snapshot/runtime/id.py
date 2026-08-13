# src/writing/snapshot/runtime/id.py
"""
B3.1: SnapshotId — Value Object for Snapshot Identity

ADR-B3-07: SnapshotId 是纯 Value Object，不承担任何存储定位职责。
"""

from dataclasses import dataclass
from uuid import UUID, uuid4


@dataclass(frozen=True)
class SnapshotId:
    """Snapshot 的唯一标识符，纯 Value Object，不绑定存储路径。"""

    _value: UUID

    @classmethod
    def new(cls) -> "SnapshotId":
        """生成一个新的随机 SnapshotId。"""
        return cls(uuid4())

    @classmethod
    def from_uuid(cls, value: UUID) -> "SnapshotId":
        """从已有的 UUID 对象创建 SnapshotId。"""
        return cls(value)

    @classmethod
    def from_string(cls, value: str) -> "SnapshotId":
        """
        从 UUID 字符串解析 SnapshotId。

        Raises:
            ValueError: 如果字符串不是合法的 UUID 格式
        """
        try:
            uuid = UUID(value)
        except ValueError as exc:
            raise ValueError("Invalid SnapshotId") from exc
        return cls(uuid)

    @property
    def value(self) -> UUID:
        """返回底层的 UUID 对象。"""
        return self._value

    def __str__(self) -> str:
        return str(self._value)