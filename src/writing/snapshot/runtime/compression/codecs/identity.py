# src/writing/snapshot/runtime/compression/codecs/identity.py
"""
B3.2/B3.5: IdentityCodec — 无压缩（默认，支持流式）
"""

from typing import Iterator

from ..protocol import CompressionCodec


class IdentityCodec:
    """无压缩编解码器（默认，支持流式透传）。"""

    id = "builtin.identity"
    display_name = "Identity (No Compression)"

    # ========== B3.2 一次性接口 ==========

    def compress(self, payload: bytes) -> bytes:
        return payload

    def decompress(self, payload: bytes) -> bytes:
        return payload

    # ========== B3.5 流式接口 ==========

    def compress_stream(self, stream: Iterator[bytes]) -> Iterator[bytes]:
        """流式透传，不进行压缩。"""
        for chunk in stream:
            yield chunk

    def decompress_stream(self, stream: Iterator[bytes]) -> Iterator[bytes]:
        """流式透传，不进行解压缩。"""
        for chunk in stream:
            yield chunk