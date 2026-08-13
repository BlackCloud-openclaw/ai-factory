# src/writing/snapshot/runtime/compression/__init__.py
"""
B3.2: Compression 模块
"""

from .errors import (
    CompressionError,
    UnsupportedCompressionError,
    CompressionDataError,
    DuplicateCompressionCodecError,
)
from .protocol import CompressionCodec
from .resolver import CompressionResolver
from .registry import CompressionRegistry
from .codecs import IdentityCodec, GzipCodec

__all__ = [
    "CompressionError",
    "UnsupportedCompressionError",
    "CompressionDataError",
    "DuplicateCompressionCodecError",
    "CompressionCodec",
    "CompressionResolver",
    "CompressionRegistry",
    "IdentityCodec",
    "GzipCodec",
]