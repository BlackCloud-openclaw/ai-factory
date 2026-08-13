# src/writing/snapshot/runtime/compression/protocol.py
"""
B3.2: CompressionCodec Protocol（无状态、纯函数）

ADR-B3-22: Codec 必须是纯函数、无状态。
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class CompressionCodec(Protocol):
    """压缩/解压缩协议。Codec 必须是无状态、纯函数的。"""

    @property
    def id(self) -> str:
        """稳定协议标识符，如 'builtin.identity'、'builtin.gzip'。"""
        ...

    @property
    def display_name(self) -> str:
        """人类可读名称（用于 UI/日志）。"""
        ...

    def compress(self, payload: bytes) -> bytes:
        """压缩字节流（纯函数）。"""
        ...

    def decompress(self, payload: bytes) -> bytes:
        """解压缩字节流（纯函数）。"""
        ...