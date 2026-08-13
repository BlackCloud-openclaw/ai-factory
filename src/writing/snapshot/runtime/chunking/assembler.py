# src/writing/snapshot/runtime/chunking/assembler.py
"""
B3.3: Assembler — 从 ChunkReader 重建字节流
"""

from typing import Iterable

from ..chunk_store.protocol import ChunkReader  # 复用 Protocol，不重新定义
from .chunk import Chunk


class Assembler:
    """从 ChunkReader 组装字节流。"""

    def __init__(self, reader: ChunkReader):
        self._reader = reader

    def assemble(self) -> bytes:
        """组装所有 Chunk 为完整字节流（一次性加载）。"""
        chunks = [self._reader.read_chunk(cid) for cid in self._reader.list_chunks()]
        return b"".join(c.payload for c in chunks)

    def assemble_stream(self) -> Iterable[bytes]:
        """流式组装，逐个返回 Chunk payload。"""
        for chunk_id in self._reader.list_chunks():
            yield self._reader.read_chunk(chunk_id).payload