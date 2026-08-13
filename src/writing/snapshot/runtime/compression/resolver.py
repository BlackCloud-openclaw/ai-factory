# src/writing/snapshot/runtime/compression/resolver.py
"""
B3.2: CompressionResolver Protocol

ADR-B3-27: Pipeline 依赖 Resolver，而非 Registry。
"""

from typing import Protocol

from .protocol import CompressionCodec


class CompressionResolver(Protocol):
    """根据 codec_id 获取 CompressionCodec。"""

    def resolve(self, codec_id: str) -> CompressionCodec:
        ...