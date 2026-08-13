# src/writing/snapshot/runtime/remote/repository.py
"""
B4: RemoteChunkRepository — 远程 ChunkRepository 实现
"""

from typing import Any, Iterator, Mapping, Union, Optional

from ..id import SnapshotId
from ..chunk_store import ChunkStore
from ..incremental import (
    ChunkSet,
    DeltaChunkSet,
    VersionManifest,
    ChunkRepository,
    VersionNotFoundError,
    VersionStore,
)
from .errors import ConcurrentModificationError, SnapshotHasChildrenError


class RemoteChunkRepository(ChunkRepository):
    """
    远程 Repository 实现，组合 ChunkStore 和 VersionStore。
    """

    def __init__(
        self,
        chunk_store: ChunkStore,
        version_store: VersionStore,
    ):
        self._chunk_store = chunk_store
        self._version_store = version_store

    # ========== B3.4 一次性接口 ==========

    def save_version(
        self,
        snapshot_id: SnapshotId,
        chunks: Union[ChunkSet, DeltaChunkSet],
        parent_id: SnapshotId | None = None,
        metadata: Mapping[str, Any] | None = None,
        *,
        expected_parent: SnapshotId | None = None,
    ) -> None:
        # 乐观锁预留（B4.7 实现）
        if expected_parent is not None:
            try:
                existing = self._version_store.get(snapshot_id)
                if existing.parent_id != expected_parent:
                    raise ConcurrentModificationError(
                        f"Expected parent {expected_parent}, got {existing.parent_id}"
                    )
            except VersionNotFoundError:
                pass

        # 存储 Chunk（使用 B3 ChunkStore）
        if isinstance(chunks, ChunkSet):
            for cid, chunk in chunks.items():
                self._chunk_store.write_chunk(snapshot_id, chunk)
        else:
            for cid, chunk in chunks.items():
                self._chunk_store.write_chunk(snapshot_id, chunk)

        # 存储 VersionManifest（使用 B3 VersionStore）
        meta = dict(metadata or {})
        if parent_id is None:
            meta["storage_mode"] = "base"
        else:
            meta["storage_mode"] = "delta"
            if isinstance(chunks, DeltaChunkSet) and chunks.deleted:
                meta["deleted"] = list(chunks.deleted)

        manifest = VersionManifest(
            snapshot_id=snapshot_id,
            parent_id=parent_id,
            metadata=meta,
        )
        self._version_store.put(manifest)

    def load_version(self, snapshot_id: SnapshotId) -> Union[ChunkSet, DeltaChunkSet]:
        manifest = self._version_store.get(snapshot_id)

        chunks_dict: dict[int, Chunk] = {}
        for cid in self._chunk_store.list_chunks(snapshot_id):
            chunks_dict[cid] = self._chunk_store.read_chunk(snapshot_id, cid)

        storage_mode = manifest.metadata.get("storage_mode", "base")
        if storage_mode == "base":
            return ChunkSet.from_mapping(chunks_dict)
        else:
            deleted = manifest.metadata.get("deleted", [])
            return DeltaChunkSet(
                added_or_modified=chunks_dict,
                deleted=frozenset(deleted),
            )

    def load_manifest(self, snapshot_id: SnapshotId) -> VersionManifest:
        return self._version_store.get(snapshot_id)

    def exists(self, snapshot_id: SnapshotId) -> bool:
        try:
            self._version_store.get(snapshot_id)
            return True
        except VersionNotFoundError:
            return False

    def delete(self, snapshot_id: SnapshotId, *, force: bool = False) -> None:
        """
        删除快照及其所有数据。

        Args:
            snapshot_id: 要删除的快照 ID
            force: 若为 False（默认），检查是否有子版本；若有则抛出 SnapshotHasChildrenError

        Raises:
            SnapshotHasChildrenError: 存在子版本且 force=False
        """
        if not force:
            for sid in self._version_store.list_ids():
                try:
                    manifest = self._version_store.get(sid)
                    if manifest.parent_id == snapshot_id:
                        raise SnapshotHasChildrenError(
                            f"Snapshot {snapshot_id} has child version {sid}. Use force=True to delete anyway."
                        )
                except VersionNotFoundError:
                    continue

        self._chunk_store.delete(snapshot_id)
        self._version_store.delete(snapshot_id)

    def list_ids(self) -> list[SnapshotId]:
        return list(self._version_store.list_ids())

    # ========== B3.5 流式接口（仅 Base） ==========

    def save_chunk_stream(
        self,
        snapshot_id: SnapshotId,
        chunks: Iterator["Chunk"],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        chunk_list = list(chunks)
        for chunk in chunk_list:
            self._chunk_store.write_chunk(snapshot_id, chunk)

        meta = dict(metadata or {})
        meta["storage_mode"] = "base"
        manifest = VersionManifest(
            snapshot_id=snapshot_id,
            parent_id=None,
            metadata=meta,
        )
        self._version_store.put(manifest)

    def load_chunk_stream(self, snapshot_id: SnapshotId) -> Iterator["Chunk"]:
        # 只有 Base 支持流式读取
        manifest = self._version_store.get(snapshot_id)
        if manifest.parent_id is not None:
            raise ValueError("Streaming load only supported for Base snapshots")

        for cid in self._chunk_store.list_chunks(snapshot_id):
            yield self._chunk_store.read_chunk(snapshot_id, cid)