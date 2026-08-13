# src/writing/snapshot/runtime/streaming/chunker.py
"""
B3.5: StreamingChunker — 流式分块器（无状态，常量内存）
"""

from typing import Iterator

from ..chunking import Chunk, ChunkingStrategy


class StreamingChunker:
    """
    流式分块器，从字节流实时生成 Chunk。

    完全无状态，每次调用 chunk_stream() 独立运行。
    当前仅支持 FixedChunkStrategy，未来可扩展。
    """

    def __init__(self, strategy: ChunkingStrategy):
        self._strategy = strategy

    def chunk_stream(self, stream: Iterator[bytes]) -> Iterator[Chunk]:
        """
        从字节流实时生成 Chunk（常量内存）。

        Args:
            stream: 字节块迭代器

        Yields:
            实时生成的 Chunk（chunk_id 从 0 开始递增）
        """
        target_size = self._get_target_size()
        if target_size is None:
            raise NotImplementedError(
                "Streaming chunking only supports FixedChunkStrategy in B3.5"
            )

        buffer = bytearray()
        chunk_id = 0

        for data in stream:
            view = memoryview(data)
            while view:
                need = target_size - len(buffer)
                if need <= 0:
                    # buffer 已满，直接输出
                    yield Chunk(chunk_id=chunk_id, payload=bytes(buffer))
                    chunk_id += 1
                    buffer = bytearray()  # 重新分配，避免引用复用
                    need = target_size

                part = view[:need]
                buffer.extend(part)
                view = view[need:]

                if len(buffer) == target_size:
                    yield Chunk(chunk_id=chunk_id, payload=bytes(buffer))
                    chunk_id += 1
                    buffer = bytearray()  # 重新分配

        # 输出剩余数据
        if buffer:
            yield Chunk(chunk_id=chunk_id, payload=bytes(buffer))

    def _get_target_size(self) -> int | None:
        """尝试从 Strategy 获取目标块大小（仅 FixedChunkStrategy）。"""
        if hasattr(self._strategy, "layout"):
            layout = self._strategy.layout
            if layout.algorithm == "fixed":
                return layout.target_chunk_size
        return None