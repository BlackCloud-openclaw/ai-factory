# src/writing/snapshot/runtime/chunking/fixed.py
"""
B3.3: FixedChunkStrategy — 固定大小分块
"""

from typing import Iterable

from .chunk import Chunk
from .strategy import ChunkingStrategy
from .layout import ChunkLayout


class FixedChunkStrategy:
    """固定大小分块策略。"""

    def __init__(self, chunk_size: int):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        self._chunk_size = chunk_size

    def chunk(self, data: bytes) -> Iterable[Chunk]:
        chunk_id = 0
        for i in range(0, len(data), self._chunk_size):
            yield Chunk(chunk_id=chunk_id, payload=data[i:i + self._chunk_size])
            chunk_id += 1

    @property
    def layout(self) -> ChunkLayout:
        return ChunkLayout(
            algorithm="fixed",
            target_chunk_size=self._chunk_size,
        )