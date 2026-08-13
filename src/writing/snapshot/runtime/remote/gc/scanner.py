# src/writing/snapshot/runtime/remote/gc/scanner.py
"""
B4.8: DeletionMarkerScanner — 扫描待处理删除标记（仅枚举）
"""

import logging
from typing import Iterator, Optional, Protocol

from ...id import SnapshotId
from ...chunk_ref import ChunkRef
from ..s3.client import S3Client
from ..s3.key_layout import S3KeyLayout
from .errors import MarkerScannerError

logger = logging.getLogger(__name__)


class DeletionMarkerScanner(Protocol):
    def iter_pending_markers(self, prefix: Optional[str] = None) -> Iterator[ChunkRef]:
        ...


class S3DeletionMarkerScanner:
    def __init__(self, client: S3Client, key_layout: S3KeyLayout):
        self._client = client
        self._key_layout = key_layout

    def iter_pending_markers(self, prefix: Optional[str] = None) -> Iterator[ChunkRef]:
        try:
            if prefix is None:
                prefix = self._key_layout.marker_prefix()

            for obj in self._client.iter_objects(prefix):
                ref = self._key_layout.parse_marker_key(obj.key)
                if ref is None:
                    logger.warning(f"Skipping malformed marker key: {obj.key}")
                    continue
                yield ref

        except MarkerScannerError:
            raise
        except Exception as e:
            raise MarkerScannerError(f"Failed to scan markers: {e}") from e