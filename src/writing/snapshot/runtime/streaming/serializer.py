# src/writing/snapshot/runtime/streaming/serializer.py
"""
B3.5: StreamingSerializer Protocol — 流式序列化接口
"""

from typing import Iterator, Protocol

from ...migration import RawSnapshot  # 修正：向上两级到 snapshot，再进入 migration


class StreamingSerializer(Protocol):
    """流式序列化协议（可选扩展，不替换 SnapshotSerializer）。"""

    def serialize_stream(self, snapshot: RawSnapshot) -> Iterator[bytes]:
        """
        流式序列化，逐块输出字节。

        Yields:
            字节块（建议 4KB-64KB），可拼接为完整序列化数据。
        """
        ...

    def deserialize_stream(self, stream: Iterator[bytes]) -> RawSnapshot:
        """
        从字节流重建 RawSnapshot。

        Args:
            stream: 字节块迭代器

        Returns:
            重建的 RawSnapshot

        Raises:
            SnapshotSerializationError: 数据不完整或格式错误
        """
        ...