# src/writing/snapshot/runtime/remote/cache/cached_repository.py
"""
B4.2: CachedChunkRepository — 完整缓存装饰器
"""

from typing import Any, Iterator, Mapping, Union, Optional

from ...id import SnapshotId
from ...incremental import (
    ChunkSet,
    DeltaChunkSet,
    VersionManifest,
    ChunkRepository,
    VersionNotFoundError,
)
from ..optimistic import OptimisticChunkRepository
from .lru_cache import LRUCache
from .metrics import CacheMetrics


class CachedChunkRepository(ChunkRepository):
    """
    Repository 缓存装饰器。

    同时实现 ChunkRepository 和 OptimisticChunkRepository。
    乐观锁能力仅在底层 remote 支持时生效，否则抛出异常。
    """

    def __init__(
        self,
        remote: ChunkRepository,
        max_entries: int = 128,
    ):
        self._remote = remote
        self._manifest_cache = LRUCache[SnapshotId, VersionManifest](max_entries)
        self._version_cache = LRUCache[SnapshotId, Union[ChunkSet, DeltaChunkSet]](max_entries)
        self._remote_reads = 0
        self._remote_writes = 0

    # ========== 公开指标 API ==========

    def metrics(self) -> dict[str, CacheMetrics]:
        return {
            "manifest": CacheMetrics(
                hits=self._manifest_cache.hits(),
                misses=self._manifest_cache.misses(),
                evictions=self._manifest_cache.evictions(),
                size=self._manifest_cache.size(),
                maxsize=self._manifest_cache.maxsize(),
                remote_reads=self._remote_reads,
                remote_writes=self._remote_writes,
            ),
            "version": CacheMetrics(
                hits=self._version_cache.hits(),
                misses=self._version_cache.misses(),
                evictions=self._version_cache.evictions(),
                size=self._version_cache.size(),
                maxsize=self._version_cache.maxsize(),
                remote_reads=self._remote_reads,
                remote_writes=self._remote_writes,
            ),
        }

    def reset_metrics(self) -> None:
        self._manifest_cache.reset_metrics()
        self._version_cache.reset_metrics()
        self._remote_reads = 0
        self._remote_writes = 0

    # ========== 内部缓存操作 ==========

    def _invalidate_all(self, snapshot_id: SnapshotId) -> None:
        self._manifest_cache.invalidate(snapshot_id)
        self._version_cache.invalidate(snapshot_id)

    def _is_optimistic(self) -> bool:
        return isinstance(self._remote, OptimisticChunkRepository)

    def _count_remote_read(self) -> None:
        self._remote_reads += 1

    def _count_remote_write(self) -> None:
        self._remote_writes += 1

    # ========== ChunkRepository 接口（B3） ==========

    def load_manifest(self, snapshot_id: SnapshotId) -> VersionManifest:
        found, cached = self._manifest_cache.lookup(snapshot_id)
        if found:
            return cached

        self._count_remote_read()
        manifest = self._remote.load_manifest(snapshot_id)
        self._manifest_cache.put(snapshot_id, manifest)
        return manifest

    def load_version(self, snapshot_id: SnapshotId) -> Union[ChunkSet, DeltaChunkSet]:
        found, cached = self._version_cache.lookup(snapshot_id)
        if found:
            return cached

        self._count_remote_read()
        # 加载 manifest 并缓存（保持一致性）
        manifest = self._remote.load_manifest(snapshot_id)
        self._manifest_cache.put(snapshot_id, manifest)
        # 加载 version 并缓存
        chunks = self._remote.load_version(snapshot_id)
        self._version_cache.put(snapshot_id, chunks)
        return chunks

    def save_version(
        self,
        snapshot_id: SnapshotId,
        chunks: Union[ChunkSet, DeltaChunkSet],
        parent_id: SnapshotId | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._count_remote_write()
        self._remote.save_version(snapshot_id, chunks, parent_id, metadata)
        self._invalidate_all(snapshot_id)

    def exists(self, snapshot_id: SnapshotId) -> bool:
        # 缓存不可信，直接查询远程（写入时已使缓存失效）
        self._count_remote_read()
        return self._remote.exists(snapshot_id)

    def delete(self, snapshot_id: SnapshotId, *, force: bool = False) -> None:
        self._count_remote_write()
        self._remote.delete(snapshot_id, force=force)
        self._invalidate_all(snapshot_id)

    def list_ids(self) -> list[SnapshotId]:
        cache_ids = set(self._manifest_cache.keys()) | set(self._version_cache.keys())
        remote_ids = set(self._remote.list_ids())
        return list(cache_ids | remote_ids)

    # ========== OptimisticChunkRepository 能力委托 ==========

    def save_version_with_expected(
        self,
        snapshot_id: SnapshotId,
        chunks: Union[ChunkSet, DeltaChunkSet],
        parent_id: SnapshotId | None = None,
        metadata: Mapping[str, Any] | None = None,
        *,
        expected_parent: SnapshotId | None = None,
    ) -> None:
        if not self._is_optimistic():
            raise TypeError(
                "Underlying repository does not support optimistic locking"
            )
        self._count_remote_write()
        self._remote.save_version(
            snapshot_id, chunks, parent_id, metadata, expected_parent=expected_parent
        )
        self._invalidate_all(snapshot_id)

    # ========== B3.5 流式接口 ==========

    def save_chunk_stream(
        self,
        snapshot_id: SnapshotId,
        chunks: Iterator["Chunk"],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._count_remote_write()
        self._remote.save_chunk_stream(snapshot_id, chunks, metadata)
        self._invalidate_all(snapshot_id)

    def load_chunk_stream(self, snapshot_id: SnapshotId) -> Iterator["Chunk"]:
        self._count_remote_read()
        for chunk in self._remote.load_chunk_stream(snapshot_id):
            yield chunk
            
    @property
    def supports_optimistic(self) -> bool:
        """检查底层 Repository 是否支持乐观锁。"""
        return hasattr(self._remote, "save_version_with_expected") or (
            hasattr(self._remote, "save_version") and
            callable(getattr(self._remote, "save_version", None))
        )

