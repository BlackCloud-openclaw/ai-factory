# src/writing/snapshot/runtime/pipeline.py
"""
B3.1/B3.2: RuntimePipeline — 协调 Serializer + Compression + Store
"""

from .id import SnapshotId
from .protocols import SnapshotStore
from .record import SnapshotRecord
from .record_builder import RecordBuilder
from .compression import CompressionResolver
from .serializers import SerializerResolver
from src.writing.snapshot.migration import RawSnapshot  # 绝对导入


class RuntimePipeline:
    def __init__(
        self,
        store: SnapshotStore,
        serializer_resolver: SerializerResolver,
        compression_resolver: CompressionResolver,
        record_builder: RecordBuilder | None = None,
    ):
        self._store = store
        self._serializer_resolver = serializer_resolver
        self._compression_resolver = compression_resolver
        self._builder = record_builder or RecordBuilder()

    def read(self, snapshot_id: SnapshotId) -> RawSnapshot:
        record = self._store.read(snapshot_id)
        codec = self._compression_resolver.resolve(record.metadata.codec_id)
        payload = codec.decompress(record.payload)
        serializer = self._serializer_resolver.resolve(record.metadata.serializer)
        return serializer.deserialize(payload)

    def write(
        self,
        snapshot_id: SnapshotId,
        snapshot: RawSnapshot,
        serializer_id: str = "builtin.json",
        codec_id: str = "builtin.identity",
    ) -> None:
        serializer = self._serializer_resolver.resolve(serializer_id)
        payload = serializer.serialize(snapshot)
        codec = self._compression_resolver.resolve(codec_id)
        compressed = codec.compress(payload)
        record = self._builder.build(
            serializer_id=serializer.id,
            codec_id=codec.id,
            content_size=len(payload),
            stored_size=len(compressed),
            payload=compressed,
        )
        self._store.write(snapshot_id, record)