# src/writing/snapshot/runtime/record_builder.py
"""
B3.2: RecordBuilder — 构建 SnapshotRecord
"""

from typing import Any, Mapping

from .metadata import SnapshotMetadata
from .record import SnapshotRecord


class RecordBuilder:
    """负责从序列化/压缩结果构建 SnapshotRecord。"""

    def build(
        self,
        serializer_id: str,
        codec_id: str,
        content_size: int,
        stored_size: int,
        payload: bytes,
        reserved: Mapping[str, Any] | None = None,
    ) -> SnapshotRecord:
        metadata = SnapshotMetadata(
            serializer=serializer_id,
            codec_id=codec_id,
            content_size=content_size,
            stored_size=stored_size,
            reserved=reserved or {},
        )
        return SnapshotRecord(metadata=metadata, payload=payload)