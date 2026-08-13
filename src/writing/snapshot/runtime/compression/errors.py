# src/writing/snapshot/runtime/compression/errors.py
"""
B3.2: Compression 错误类型
"""

from ..exceptions import SnapshotRuntimeError


class CompressionError(SnapshotRuntimeError):
    """压缩/解压缩基类异常。"""
    pass


class UnsupportedCompressionError(CompressionError):
    """未知的压缩算法 ID。"""
    pass


class CompressionDataError(CompressionError):
    """压缩数据损坏或不完整。"""
    pass


class DuplicateCompressionCodecError(CompressionError):
    """尝试重复注册同名 Codec。"""
    pass