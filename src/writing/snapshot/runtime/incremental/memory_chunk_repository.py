# src/writing/snapshot/runtime/incremental/memory_chunk_repository.py
"""
B3.4/B3.5: MemoryChunkRepository — 内存存储实现（测试用）
"""

from typing import Any, Iterator, Mapping, Union

from ..id import SnapshotId
from .chunk_set import ChunkSet
from .delta_chunk_set import DeltaChunkSet
from .version_manifest import VersionManifest
from .version_errors import VersionNotFoundError


class MemoryChunkRepository:
    """内存实现的 ChunkRepository（仅供测试）。"""

    def __init__(self):
        self._chunks: dict[SnapshotId, Union[ChunkSet, DeltaChunkSet]] = {}
        self._manifests: dict[SnapshotId, VersionManifest] = {}

    # ========== B3.4 一次性接口 ==========

    def save_version(
        self,
        snapshot_id: SnapshotId,
        chunks: Union[ChunkSet, DeltaChunkSet],
        parent_id: SnapshotId | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(chunks, ChunkSet):
            stored_chunks = ChunkSet.from_mapping(dict(chunks.items()))
        else:
            stored_chunks = DeltaChunkSet(
                added_or_modified=dict(chunks.items()),
                deleted=chunks.deleted,
            )

        self._chunks[snapshot_id] = stored_chunks

        meta = dict(metadata or {})
        if parent_id is None:
            meta["storage_mode"] = "base"
        else:
            meta["storage_mode"] = "delta"

        manifest = VersionManifest(
            snapshot_id=snapshot_id,
            parent_id=parent_id,
            metadata=meta,
        )
        self._manifests[snapshot_id] = manifest

    def load_version(self, snapshot_id: SnapshotId) -> Union[ChunkSet, DeltaChunkSet]:
        if snapshot_id not in self._chunks:
            raise VersionNotFoundError(f"Version not found: {snapshot_id}")
        return self._chunks[snapshot_id]

    def load_manifest(self, snapshot_id: SnapshotId) -> VersionManifest:
        if snapshot_id not in self._manifests:
            raise VersionNotFoundError(f"Manifest not found: {snapshot_id}")
        return self._manifests[snapshot_id]

    def exists(self, snapshot_id: SnapshotId) -> bool:
        return snapshot_id in self._manifests

    def delete(self, snapshot_id: SnapshotId) -> None:
        self._chunks.pop(snapshot_id, None)
        self._manifests.pop(snapshot_id, None)

    def list_ids(self) -> list[SnapshotId]:
        return list(self._manifests.keys())

    # ========== B3.5 流式接口（仅 Base） ==========

    def save_chunk_stream(
        self,
        snapshot_id: SnapshotId,
        chunks: Iterator["Chunk"],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """
        保存完整 Chunk 流（仅用于 Base Snapshot）。

        NOTE:
        Memory implementation intentionally buffers all chunks.
        Production repository implementations should persist incrementally.
        """
        chunk_list = list(chunks)
        chunk_set = ChunkSet.from_mapping({c.chunk_id: c for c in chunk_list})
        self.save_version(
            snapshot_id=snapshot_id,
            chunks=chunk_set,
            parent_id=None,
            metadata=metadata,
        )

    def load_chunk_stream(self, snapshot_id: SnapshotId) -> Iterator["Chunk"]:
        """
        流式加载 Chunk 序列。

        NOTE:
        Memory implementation loads all chunks at once.
        Production implementations should stream from underlying storage.
        """
        chunk_data = self.load_version(snapshot_id)
        if isinstance(chunk_data, ChunkSet):
            for cid in sorted(chunk_data.keys()):
                chunk = chunk_data.get(cid)
                if chunk is not None:
                    yield chunk
        elif isinstance(chunk_data, DeltaChunkSet):
            for cid in sorted(chunk_data.keys()):
                chunk = chunk_data.added_or_modified.get(cid)
                if chunk is not None:
                    yield chunk
        else:
            raise TypeError(f"Unexpected chunk data type: {type(chunk_data)}")