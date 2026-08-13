# src/writing/snapshot/migration/raw_snapshot.py

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, FrozenSet

from .deep_freeze import deep_freeze
from .version import SchemaVersion


@dataclass(frozen=True)
class RawSnapshot(Mapping[str, Any]):
    """Snapshot 的只读不可变视图，作为 Migration Runtime 的统一输入。"""

    schema_version: SchemaVersion
    _data: Mapping[str, Any] = field(repr=False)
    _fields: FrozenSet[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        frozen = deep_freeze(self._data)
        object.__setattr__(self, "_data", frozen)
        object.__setattr__(self, "_fields", frozenset(frozen.keys()))

    @classmethod
    def from_mapping(
        cls,
        *,
        schema_version: SchemaVersion,
        data: Mapping[str, Any],
    ) -> RawSnapshot:
        return cls(schema_version=schema_version, _data=data)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def require(self, key: str) -> Any:
        if key not in self._data:
            raise KeyError(f"Required field '{key}' not found in snapshot")
        return self._data[key]

    def has_field(self, key: str) -> bool:
        return key in self._data

    def fields(self) -> FrozenSet[str]:
        return self._fields

    # Mapping Protocol
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._data

    def to_mapping(self) -> Mapping[str, Any]:
        """
        返回内部数据的只读视图。

        返回的 Mapping 是不可变的（MappingProxyType），
        任何修改尝试都会抛出 TypeError。

        此方法是 Loader 获取 RawSnapshot 数据的唯一公开 API，
        请勿直接访问 _data 私有字段。
        """
        return self._data