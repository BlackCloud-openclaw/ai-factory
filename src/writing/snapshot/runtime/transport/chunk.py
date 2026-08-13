# src/writing/snapshot/runtime/transport/chunk.py
"""
B3.3: ChunkTransport — 分块传输（Chunk Transport）

注意：当前实现仍一次性处理整个数据，真正的 Streaming 将在 B3.5 实现。
"""

from ...migration import RawSnapshot
from ..id import SnapshotId
from ..compression import CompressionResolver
from ..serializers import SerializerResolver
from ..chunking import ChunkingStrategy, FixedChunkStrategy, ManifestBuilder, Assembler
from ..chunk_store import ChunkStore
from .protocol import Transport


class ChunkTransport:
    """
    分块传输。

    将快照分块存储，每个 Chunk 独立存储。
    """

    def __init__(
        self,
        store: ChunkStore,
        serializer_resolver: SerializerResolver,
        compression_resolver: CompressionResolver,
        strategy: ChunkingStrategy | None = None,
        default_serializer_id: str = "builtin.json",
        default_codec_id: str = "builtin.identity",
    ):
        self._store = store
        self._serializer_resolver = serializer_resolver
        self._compression_resolver = compression_resolver
        self._strategy = strategy or FixedChunkStrategy(1024 * 1024)
        self._default_serializer_id = default_serializer_id
        self._default_codec_id = default_codec_id

    def write(self, snapshot_id: SnapshotId, snapshot: RawSnapshot) -> None:
        # 1. 序列化
        serializer = self._serializer_resolver.resolve(self._default_serializer_id)
        payload = serializer.serialize(snapshot)

        # 2. 压缩（一次性）
        codec = self._compression_resolver.resolve(self._default_codec_id)
        compressed = codec.compress(payload)

        # 3. 分块
        writer = self._store.create_writer(snapshot_id)
        builder = ManifestBuilder(
            snapshot_id=snapshot_id,
            serializer_id=self._default_serializer_id,
            codec_id=self._default_codec_id,
            layout=self._strategy.layout,
        )

        for chunk in self._strategy.chunk(compressed):
            writer.append(chunk)
            builder.record_chunk(len(chunk.payload))

        # 4. 写入 Manifest
        manifest = builder.build(total_size=len(payload))
        self._store.write_manifest(snapshot_id, manifest)

    def read(self, snapshot_id: SnapshotId) -> RawSnapshot:
        # 1. 读取 Manifest
        manifest = self._store.read_manifest(snapshot_id)

        # 2. 读取 Chunk 并组装
        reader = self._store.create_reader(snapshot_id)
        assembler = Assembler(reader)
        compressed = assembler.assemble()

        # 3. 解压缩
        codec = self._compression_resolver.resolve(manifest.codec_id)
        payload = codec.decompress(compressed)

        # 4. 反序列化
        serializer = self._serializer_resolver.resolve(manifest.serializer_id)
        return serializer.deserialize(payload)