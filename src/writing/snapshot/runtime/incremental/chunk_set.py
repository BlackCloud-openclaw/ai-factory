# src/writing/snapshot/runtime/incremental/chunk_set.py
"""
B3.4: ChunkSet — 不可变完整快照的 Chunk 集合
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import ItemsView, KeysView, Mapping, ValuesView, Iterator

from ..chunking import Chunk


@dataclass(frozen=True)
class ChunkSet:
    """
    完整快照的 Chunk 集合（不可变值对象）。

    内部使用 MappingProxyType 保证不可变。
    对外提供视图接口，不暴露底层 dict。
    不提供排序，排序由调用方决定。
    """

    _chunks: Mapping[int, Chunk] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self._chunks, MappingProxyType):
            object.__setattr__(self, "_chunks", MappingProxyType(dict(self._chunks)))

    def get(self, chunk_id: int) -> Chunk | None:
        """获取指定 chunk_id 的 Chunk，不存在时返回 None。"""
        return self._chunks.get(chunk_id)

    def keys(self) -> KeysView[int]:
        """返回所有 chunk_id 的视图（不排序）。"""
        return self._chunks.keys()

    def values(self) -> ValuesView[Chunk]:
        """返回所有 Chunk 的视图（不排序）。"""
        return self._chunks.values()

    def items(self) -> ItemsView[int, Chunk]:
        """返回所有 (chunk_id, Chunk) 的视图（不排序）。"""
        return self._chunks.items()

    def __len__(self) -> int:
        return len(self._chunks)

    def __iter__(self) -> Iterator[int]:
        """迭代 chunk_id（等价于 keys()）。"""
        return iter(self._chunks)

    def __contains__(self, chunk_id: object) -> bool:
        """支持 'chunk_id in chunk_set' 语法。"""
        return isinstance(chunk_id, int) and chunk_id in self._chunks

    @classmethod
    def from_mapping(cls, chunks: Mapping[int, Chunk]) -> ChunkSet:
        """从 Mapping 构造（自动冻结）。"""
        return cls(_chunks=chunks)

    @classmethod
    def empty(cls) -> ChunkSet:
        """返回空集合。"""
        return cls()


# 模块级共享空常量（安全、只读）
EMPTY_CHUNK_SET = ChunkSet.empty()