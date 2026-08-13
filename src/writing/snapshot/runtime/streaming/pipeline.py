# src/writing/snapshot/runtime/streaming/pipeline.py
"""
B3.5: StreamingPipeline — 流式处理流水线
"""

from typing import Iterator

from ...migration import RawSnapshot  # 修正
from ..serializers import JsonSerializer
from ..compression import GzipCodec
from ..chunking import ChunkingStrategy, FixedChunkStrategy, Chunk
from .serializer import StreamingSerializer
from .compression import StreamingCompressionCodec
from .chunker import StreamingChunker


class StreamingPipeline:
    """
    流式处理流水线。

    将 RawSnapshot 流式转换为 Chunk 序列，支持常量内存处理。
    """

    def __init__(
        self,
        serializer: StreamingSerializer,
        compression: StreamingCompressionCodec,
        strategy: ChunkingStrategy,
    ):
        self._serializer = serializer
        self._compression = compression
        self._chunker = StreamingChunker(strategy)

    def write_stream(
        self,
        snapshot: RawSnapshot,
    ) -> Iterator[Chunk]:
        """
        将 RawSnapshot 流式转换为 Chunk 序列。

        Yields:
            Chunk 序列（内存 O(1)）
        """
        # 流式序列化
        bytes_stream = self._serializer.serialize_stream(snapshot)

        # 流式压缩
        compressed_stream = self._compression.compress_stream(bytes_stream)

        # 流式分块
        for chunk in self._chunker.chunk_stream(compressed_stream):
            yield chunk

    def read_stream(
        self,
        chunks: Iterator[Chunk],
    ) -> RawSnapshot:
        """
        从 Chunk 序列流式重建 RawSnapshot。

        Args:
            chunks: Chunk 迭代器

        Returns:
            重建的 RawSnapshot
        """
        # 组装字节流
        bytes_stream = (chunk.payload for chunk in chunks)

        # 流式解压缩
        decompressed_stream = self._compression.decompress_stream(bytes_stream)

        # 流式反序列化
        return self._serializer.deserialize_stream(decompressed_stream)


def create_default_streaming_pipeline() -> StreamingPipeline:
    """创建默认流式流水线（JsonSerializer + GzipCodec + FixedChunkStrategy）。"""
    return StreamingPipeline(
        serializer=JsonSerializer(),
        compression=GzipCodec(),
        strategy=FixedChunkStrategy(1024 * 1024),
    )