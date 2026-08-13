# src/writing/snapshot/runtime/chunking/strategy.py
"""
B3.3: ChunkingStrategy Protocol — 分块算法协议
"""

from typing import Iterable, Protocol

from .chunk import Chunk
from .layout import ChunkLayout


class ChunkingStrategy(Protocol):
    """分块算法协议。"""

    def chunk(self, data: bytes) -> Iterable[Chunk]:
        """将数据切分为 Chunk 序列。"""
        ...

    @property
    def layout(self) -> ChunkLayout:
        """返回该策略的布局描述，供 Manifest 使用。"""
        ...