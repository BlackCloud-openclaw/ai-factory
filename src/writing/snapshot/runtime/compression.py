# src/writing/snapshot/runtime/compression.py
"""
B3.2: Compression 预留 (B3.1 仅提供协议占位)
"""

from typing import Protocol


class CompressionCodec(Protocol):
    """压缩/解压缩协议。"""

    name: str

    def compress(self, data: bytes) -> bytes:
        ...

    def decompress(self, data: bytes) -> bytes:
        ...


class CompressionRegistry:
    """压缩编解码器注册表。"""

    _codecs: dict[str, CompressionCodec] = {}

    @classmethod
    def register(cls, codec: CompressionCodec) -> None:
        cls._codecs[codec.name] = codec

    @classmethod
    def get(cls, name: str) -> CompressionCodec:
        if name not in cls._codecs:
            raise ValueError(f"Unknown compression: {name}")
        return cls._codecs[name]


class IdentityCodec:
    """无压缩编解码器（默认）。"""

    name = "identity"

    def compress(self, data: bytes) -> bytes:
        return data

    def decompress(self, data: bytes) -> bytes:
        return data