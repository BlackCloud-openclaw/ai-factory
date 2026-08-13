# src/writing/snapshot/runtime/streaming/compression.py
"""
B3.5: StreamingCompressionCodec Protocol — 流式压缩接口
"""

from typing import Iterator, Protocol


class StreamingCompressionCodec(Protocol):
    """流式压缩协议（可选扩展）。"""

    def compress_stream(self, stream: Iterator[bytes]) -> Iterator[bytes]:
        """
        流式压缩输入字节流。

        Args:
            stream: 原始字节块迭代器

        Yields:
            压缩后的字节块
        """
        ...

    def decompress_stream(self, stream: Iterator[bytes]) -> Iterator[bytes]:
        """
        流式解压缩输入字节流。

        Args:
            stream: 压缩字节块迭代器

        Yields:
            解压后的字节块
        """
        ...