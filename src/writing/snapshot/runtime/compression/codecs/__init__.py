# src/writing/snapshot/runtime/compression/codecs/__init__.py
"""
B3.2: 内置压缩编解码器
"""

from .identity import IdentityCodec
from .gzip import GzipCodec

__all__ = [
    "IdentityCodec",
    "GzipCodec",
]