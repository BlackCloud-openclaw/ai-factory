# src/writing/snapshot/runtime/compression/registry.py
"""
B3.2: CompressionRegistry — 压缩编解码器注册表（不可变）

ADR-B3-23: Registry 不负责实例化，仅负责管理。
ADR-B3-26: Registry 禁止重复注册。
"""

from .errors import DuplicateCompressionCodecError, UnsupportedCompressionError
from .protocol import CompressionCodec
from .resolver import CompressionResolver


class CompressionRegistry(CompressionResolver):
    """压缩编解码器注册表（不可变，构造时注入）。"""

    def __init__(self, codecs: list[CompressionCodec] | None = None):
        self._codecs: dict[str, CompressionCodec] = {}
        if codecs:
            for codec in codecs:
                if codec.id in self._codecs:
                    raise DuplicateCompressionCodecError(
                        f"Codec with id '{codec.id}' already registered"
                    )
                self._codecs[codec.id] = codec

    def resolve(self, codec_id: str) -> CompressionCodec:
        """实现 CompressionResolver 协议。"""
        if codec_id not in self._codecs:
            raise UnsupportedCompressionError(f"Unknown compression: {codec_id}")
        return self._codecs[codec_id]

    def list(self) -> list[str]:
        """列出所有已注册的压缩算法 ID。"""
        return list(self._codecs.keys())

    @classmethod
    def with_builtin(cls) -> "CompressionRegistry":
        """创建包含所有内置 Codec 的 Registry。"""
        from .codecs import IdentityCodec, GzipCodec

        return cls([IdentityCodec(), GzipCodec()])