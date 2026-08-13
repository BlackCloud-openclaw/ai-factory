# src/writing/snapshot/runtime/transport/record.py
"""
B3.3: RecordTransport — 单记录传输（B3.1 逻辑迁移）
"""

from ...migration import RawSnapshot
from ..id import SnapshotId
from ..metadata import SnapshotMetadata
from ..protocols import SnapshotStore
from ..record import SnapshotRecord
from ..compression import CompressionResolver
from ..serializers import SerializerResolver
from .protocol import Transport


class RecordTransport:
    """
    单记录传输（默认）。

    将整个快照序列化为单条 SnapshotRecord。
    """

    def __init__(
        self,
        store: SnapshotStore,
        serializer_resolver: SerializerResolver,
        compression_resolver: CompressionResolver,
        default_serializer_id: str = "builtin.json",
        default_codec_id: str = "builtin.identity",
    ):
        self._store = store
        self._serializer_resolver = serializer_resolver
        self._compression_resolver = compression_resolver
        self._default_serializer_id = default_serializer_id
        self._default_codec_id = default_codec_id

    def write(self, snapshot_id: SnapshotId, snapshot: RawSnapshot) -> None:
        serializer = self._serializer_resolver.resolve(self._default_serializer_id)
        payload = serializer.serialize(snapshot)

        codec = self._compression_resolver.resolve(self._default_codec_id)
        compressed = codec.compress(payload)

        metadata = SnapshotMetadata(
            serializer=self._default_serializer_id,
            codec_id=self._default_codec_id,
            content_size=len(payload),
            stored_size=len(compressed),
        )
        record = SnapshotRecord(metadata=metadata, payload=compressed)
        self._store.write(snapshot_id, record)

    def read(self, snapshot_id: SnapshotId) -> RawSnapshot:
        record = self._store.read(snapshot_id)

        codec = self._compression_resolver.resolve(record.metadata.codec_id)
        payload = codec.decompress(record.payload)

        serializer = self._serializer_resolver.resolve(record.metadata.serializer)
        return serializer.deserialize(payload)