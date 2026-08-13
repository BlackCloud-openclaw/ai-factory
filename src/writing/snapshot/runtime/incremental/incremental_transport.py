# src/writing/snapshot/runtime/incremental/incremental_transport.py
"""
B3.4/B3.5: IncrementalTransport — 增量快照传输层（支持流式 Base）
"""

from typing import Optional, Iterator

from ..id import SnapshotId
from ..transport import Transport
from ..serializers import SerializerResolver
from ..compression import CompressionResolver
from ..chunking import ChunkingStrategy, FixedChunkStrategy, Chunk
from ..streaming import StreamingPipeline
from .chunk_set import ChunkSet
from .delta_chunk_set import DeltaChunkSet
from .delta_calculator import DeltaCalculator
from .chunk_repository import ChunkRepository
from .version_resolver import VersionResolver
from .version_errors import VersionNotFoundError
from ...migration import RawSnapshot


class IncrementalTransport(Transport):
    """
    增量快照传输层，支持流式 Base 写入（B3.5）和缓冲写入（B3.4）。

    流式路径仅支持 Base（无 parent）的写入和读取；Delta 链仍使用缓冲路径。
    """

    def __init__(
        self,
        repository: ChunkRepository,
        serializer_resolver: SerializerResolver,
        compression_resolver: CompressionResolver,
        strategy: ChunkingStrategy | None = None,
        default_serializer_id: str = "builtin.json",
        default_codec_id: str = "builtin.identity",
        max_chain_depth: int = 32,
        streaming_pipeline: Optional[StreamingPipeline] = None,
    ):
        self._repository = repository
        self._resolver = VersionResolver(repository, max_depth=max_chain_depth)
        self._serializer_resolver = serializer_resolver
        self._compression_resolver = compression_resolver
        self._strategy = strategy or FixedChunkStrategy(1024 * 1024)
        self._default_serializer_id = default_serializer_id
        self._default_codec_id = default_codec_id

        self._streaming_pipeline = streaming_pipeline
        self._streaming_supported = self._check_streaming_support()

    def _check_streaming_support(self) -> bool:
        """检查所有组件是否支持流式处理（写入 + 读取）。"""
        # 1. Serializer 流式支持
        serializer = self._serializer_resolver.resolve(self._default_serializer_id)
        if not callable(getattr(serializer, "serialize_stream", None)):
            return False
        if not callable(getattr(serializer, "deserialize_stream", None)):
            return False

        # 2. Compression 流式支持
        codec = self._compression_resolver.resolve(self._default_codec_id)
        if not callable(getattr(codec, "compress_stream", None)):
            return False
        if not callable(getattr(codec, "decompress_stream", None)):
            return False

        # 3. Repository 流式支持
        if not callable(getattr(self._repository, "save_chunk_stream", None)):
            return False
        if not callable(getattr(self._repository, "load_chunk_stream", None)):
            return False

        # 4. 构造 Pipeline（如果未提供）
        if self._streaming_pipeline is None:
            from ..streaming import StreamingPipeline
            self._streaming_pipeline = StreamingPipeline(
                serializer=serializer,
                compression=codec,
                strategy=self._strategy,
            )

        return True

    def write(self, snapshot_id: SnapshotId, snapshot: RawSnapshot) -> None:
        if self._repository.exists(snapshot_id):
            self._repository.delete(snapshot_id)

        parent_id = self._get_latest_version()

        if parent_id is None and self._streaming_supported:
            self._write_streaming(snapshot_id, snapshot)
        else:
            self._write_buffered(snapshot_id, snapshot, parent_id)

    def read(self, snapshot_id: SnapshotId) -> RawSnapshot:
        try:
            manifest = self._repository.load_manifest(snapshot_id)
            is_base = manifest.parent_id is None
        except VersionNotFoundError:
            raise VersionNotFoundError(f"Version not found: {snapshot_id}")

        if is_base and self._streaming_supported:
            return self._read_streaming(snapshot_id)
        else:
            return self._read_buffered(snapshot_id)

    def _write_streaming(self, snapshot_id: SnapshotId, snapshot: RawSnapshot) -> None:
        chunks = self._streaming_pipeline.write_stream(snapshot)
        self._repository.save_chunk_stream(
            snapshot_id=snapshot_id,
            chunks=chunks,
            metadata={
                "compression": self._default_codec_id,
                "serializer": self._default_serializer_id,
            },
        )

    def _write_buffered(self, snapshot_id: SnapshotId, snapshot: RawSnapshot, parent_id: Optional[SnapshotId]) -> None:
        chunk_set = self._snapshot_to_chunk_set(snapshot)

        if parent_id is not None:
            parent_chunk_set = self._resolve_chunk_set(parent_id)
            delta = DeltaCalculator.compute_delta(parent_chunk_set, chunk_set)
            self._repository.save_version(
                snapshot_id=snapshot_id,
                chunks=delta,
                parent_id=parent_id,
                metadata={
                    "compression": self._default_codec_id,
                    "serializer": self._default_serializer_id,
                },
            )
        else:
            self._repository.save_version(
                snapshot_id=snapshot_id,
                chunks=chunk_set,
                parent_id=None,
                metadata={
                    "compression": self._default_codec_id,
                    "serializer": self._default_serializer_id,
                },
            )

    def _read_streaming(self, snapshot_id: SnapshotId) -> RawSnapshot:
        chunks = self._repository.load_chunk_stream(snapshot_id)
        return self._streaming_pipeline.read_stream(chunks)

    def _read_buffered(self, snapshot_id: SnapshotId) -> RawSnapshot:
        chain = self._resolver.resolve_chain(snapshot_id)

        base = self._repository.load_version(chain.base)
        if not isinstance(base, ChunkSet):
            raise TypeError(f"Expected ChunkSet for base, got {type(base)}")

        current = base
        for version in chain.versions[1:]:
            delta = self._repository.load_version(version)
            if not isinstance(delta, DeltaChunkSet):
                raise TypeError(f"Expected DeltaChunkSet, got {type(delta)}")
            current = DeltaCalculator.apply_delta(current, delta)

        return self._chunk_set_to_snapshot(current)

    def _get_latest_version(self) -> Optional[SnapshotId]:
        all_ids = set(self._repository.list_ids())
        if not all_ids:
            return None

        children = set()
        for sid in all_ids:
            try:
                manifest = self._repository.load_manifest(sid)
                if manifest.parent_id is not None:
                    children.add(manifest.parent_id)
            except VersionNotFoundError:
                continue

        latest = [sid for sid in all_ids if sid not in children]
        if not latest:
            return None
        return latest[0]

    def _resolve_chunk_set(self, snapshot_id: SnapshotId) -> ChunkSet:
        chain = self._resolver.resolve_chain(snapshot_id)

        base = self._repository.load_version(chain.base)
        if not isinstance(base, ChunkSet):
            raise TypeError(f"Expected ChunkSet for base, got {type(base)}")

        current = base
        for version in chain.versions[1:]:
            delta = self._repository.load_version(version)
            if not isinstance(delta, DeltaChunkSet):
                raise TypeError(f"Expected DeltaChunkSet, got {type(delta)}")
            current = DeltaCalculator.apply_delta(current, delta)

        return current

    def _snapshot_to_chunk_set(self, snapshot: RawSnapshot) -> ChunkSet:
        serializer = self._serializer_resolver.resolve(self._default_serializer_id)
        payload = serializer.serialize(snapshot)

        codec = self._compression_resolver.resolve(self._default_codec_id)
        compressed = codec.compress(payload)

        chunks: dict[int, Chunk] = {}
        for chunk in self._strategy.chunk(compressed):
            chunks[chunk.chunk_id] = chunk

        return ChunkSet.from_mapping(chunks)

    def _chunk_set_to_snapshot(self, chunk_set: ChunkSet) -> RawSnapshot:
        compressed = b"".join(
            chunk_set.get(cid).payload for cid in sorted(chunk_set.keys())
        )
        codec = self._compression_resolver.resolve(self._default_codec_id)
        payload = codec.decompress(compressed)

        serializer = self._serializer_resolver.resolve(self._default_serializer_id)
        return serializer.deserialize(payload)