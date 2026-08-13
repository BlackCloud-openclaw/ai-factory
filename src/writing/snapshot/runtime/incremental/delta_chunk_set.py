# src/writing/snapshot/runtime/incremental/delta_chunk_set.py
"""
B3.4: DeltaChunkSet — 相对于父版本的差异
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import FrozenSet, ItemsView, KeysView, Mapping, ValuesView, Iterator

from ..chunking import Chunk  # Blocking 修正：导入 Chunk


@dataclass(frozen=True)
class DeltaChunkSet:
    """
    增量快照：相对于父版本的 Chunk 变更集合。

    不变量：
        added_or_modified 与 deleted 不相交。
    """

    added_or_modified: Mapping[int, Chunk] = field(default_factory=dict)
    deleted: FrozenSet[int] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not isinstance(self.added_or_modified, MappingProxyType):
            object.__setattr__(
                self,
                "added_or_modified",
                MappingProxyType(dict(self.added_or_modified)),
            )
        if not isinstance(self.deleted, frozenset):
            object.__setattr__(self, "deleted", frozenset(self.deleted))

        # 不变量：不相交
        added_keys = set(self.added_or_modified.keys())
        deleted_set = set(self.deleted)
        intersection = added_keys & deleted_set
        if intersection:
            raise ValueError(
                f"DeltaChunkSet invariant violated: "
                f"chunk(s) {intersection} appear in both added_or_modified and deleted."
            )

    def is_empty(self) -> bool:
        return not self.added_or_modified and not self.deleted

    def keys(self) -> KeysView[int]:
        return self.added_or_modified.keys()

    def values(self) -> ValuesView[Chunk]:
        return self.added_or_modified.values()

    def items(self) -> ItemsView[int, Chunk]:
        return self.added_or_modified.items()

    def deleted_keys(self) -> FrozenSet[int]:
        return self.deleted

    def __len__(self) -> int:
        return len(self.added_or_modified)

    def __iter__(self) -> Iterator[int]:
        """迭代新增/修改的 chunk_id（等价于 keys()）。"""
        return iter(self.added_or_modified)

    @classmethod
    def empty(cls) -> DeltaChunkSet:
        return cls()