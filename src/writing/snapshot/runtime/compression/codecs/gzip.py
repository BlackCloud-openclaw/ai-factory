# src/writing/snapshot/runtime/compression/codecs/gzip.py
"""
B3.2/B3.5: GzipCodec — gzip 压缩（确定性输出 + 流式支持）
"""

import gzip
import zlib
from io import BytesIO
from typing import Iterator

from ..errors import CompressionDataError
from ..protocol import CompressionCodec


class GzipCodec:
    """gzip 压缩编解码器（确定性输出，完全无状态，支持流式）。"""

    id = "builtin.gzip"
    display_name = "Gzip"

    def __init__(self, compresslevel: int = 6):
        self._compresslevel = compresslevel

    def compress(self, payload: bytes) -> bytes:
        """一次性压缩（B3.2 兼容）。"""
        out = BytesIO()
        with gzip.GzipFile(
            fileobj=out,
            mode="wb",
            compresslevel=self._compresslevel,
            mtime=0,
        ) as f:
            f.write(payload)
        return out.getvalue()

    def decompress(self, payload: bytes) -> bytes:
        """一次性解压缩（B3.2 兼容）。"""
        try:
            return gzip.decompress(payload)
        except (gzip.BadGzipFile, EOFError, OSError) as e:
            raise CompressionDataError(f"Gzip decompression failed: {e}") from e

    # ========== B3.5: 流式方法 ==========

    def compress_stream(self, stream: Iterator[bytes]) -> Iterator[bytes]:
        """
        流式 gzip 压缩（常量内存）。

        使用 zlib.compressobj 实现真正的流式压缩。
        """
        compressobj = zlib.compressobj(
            level=self._compresslevel,
            wbits=zlib.MAX_WBITS | 16,  # gzip 格式
        )

        for chunk in stream:
            if chunk:
                compressed = compressobj.compress(chunk)
                if compressed:
                    yield compressed

        remaining = compressobj.flush()
        if remaining:
            yield remaining

    def decompress_stream(self, stream: Iterator[bytes]) -> Iterator[bytes]:
        """
        流式 gzip 解压缩（常量内存）。

        使用 zlib.decompressobj 实现流式解压。
        """
        decompressobj = zlib.decompressobj(
            wbits=zlib.MAX_WBITS | 16,  # gzip 格式
        )

        for chunk in stream:
            if chunk:
                decompressed = decompressobj.decompress(chunk)
                if decompressed:
                    yield decompressed

        remaining = decompressobj.flush()
        if remaining:
            yield remaining