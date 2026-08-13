# src/writing/snapshot/runtime/remote/gc/collector.py
"""
B4.5/B4.6/B4.7.1: GarbageCollector — 垃圾回收执行器（安全删除版）
"""

import os
import time
import uuid
from typing import Optional, Set, List, Dict
from datetime import datetime

from ...incremental import VersionStore
from ...id import SnapshotId
from ...chunk_ref import ChunkRef
from .models import (
    ChunkMetadata,
    GCResult,
    GCStats,
    DeletionCandidate,
)
from .deletion import DeletionPlanner, DeletionPlan
from .reachability import ReachabilityAnalyzer
from .retention import RetentionPolicy, KeepAllPolicy
from .errors import (
    GarbageCollectionError,
    GCNotSupportedError,
    GracePeriodNotElapsedError,
    LeaseConflictError,
    LeaseAcquisitionError,
    DeletionFailedError,
    DeletionMarkerError,
)
from .capability import ChunkEnumerator, GCDeleteAdapter, ChunkMetadataProvider
from .lease import LeaseManager
from .deletion_marker_store import DeletionMarkerStore


class GarbageCollector:
    """
    垃圾回收执行器（支持 ChunkMetadata + Lease + Grace Period）。
    所有安全控制作用于 Chunk 级别。
    """

    def __init__(
        self,
        version_store: VersionStore,
        chunk_enumerator: ChunkEnumerator,
        delete_adapter: Optional[GCDeleteAdapter] = None,
        metadata_provider: Optional[ChunkMetadataProvider] = None,
        retention_policy: Optional[RetentionPolicy] = None,
        *,
        batch_size: int = 100,
        lease_manager: Optional[LeaseManager] = None,
        deletion_marker_store: Optional[DeletionMarkerStore] = None,
        grace_period_seconds: int = 86400,  # 1 day
        lease_ttl_seconds: int = 300,       # 5 minutes
        owner_id: Optional[str] = None,
    ):
        self._version_store = version_store
        self._chunk_enumerator = chunk_enumerator
        self._delete_adapter = delete_adapter
        self._metadata_provider = metadata_provider
        self._retention_policy = retention_policy or KeepAllPolicy()
        self._batch_size = batch_size
        self._lease_manager = lease_manager
        self._deletion_marker_store = deletion_marker_store
        self._grace_period_seconds = grace_period_seconds
        self._lease_ttl_seconds = lease_ttl_seconds
        self._owner_id = owner_id or f"gc-{os.uname().nodename}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._stats = GCStats()

    def collect(
        self,
        *,
        dry_run: bool = True,
        force: bool = False,  # 是否跳过所有安全检查（仅紧急恢复）
    ) -> GCResult:
        start_time = time.perf_counter()

        try:
            # 1. 分析可达性
            analyzer = ReachabilityAnalyzer(
                self._version_store,
                self._chunk_enumerator,
                self._retention_policy,
            )
            graph = analyzer.analyze()

            # 2. 获取所有物理 Chunk 并获取元数据
            all_physical_chunks = self._collect_all_physical_chunks_with_metadata()

            # 3. 计算待删除的 Chunk
            planner = DeletionPlanner()
            plan = planner.plan(all_physical_chunks, graph.reachable_chunks)

            # 4. 执行删除
            deleted_chunks: set[ChunkRef] = set()
            reclaimed_bytes: int = 0

            if dry_run:
                deleted_chunks = {c.chunk_ref for c in plan.candidates}
                reclaimed_bytes = sum(c.size_bytes for c in plan.candidates)
            else:
                if self._delete_adapter is None:
                    raise GCNotSupportedError(
                        "Delete adapter is not available. "
                        "Provide a GCDeleteAdapter to enable actual deletion."
                    )
                deleted_chunks, reclaimed_bytes = self._execute_deletion_with_safety(
                    plan.candidates, force
                )

            duration_ms = int((time.perf_counter() - start_time) * 1000)
            self._stats = GCStats(
                total_runs=self._stats.total_runs + 1,
                total_deleted_chunks=self._stats.total_deleted_chunks + len(deleted_chunks),
                total_reclaimed_bytes=self._stats.total_reclaimed_bytes + reclaimed_bytes,
                total_duration_ms=self._stats.total_duration_ms + duration_ms,
                last_run=datetime.now(),
            )

            return GCResult(
                deleted_chunks=frozenset(deleted_chunks),
                reclaimed_bytes=reclaimed_bytes,
                dry_run=dry_run,
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            self._stats = GCStats(
                total_runs=self._stats.total_runs + 1,
                total_deleted_chunks=self._stats.total_deleted_chunks,
                total_reclaimed_bytes=self._stats.total_reclaimed_bytes,
                total_duration_ms=self._stats.total_duration_ms + duration_ms,
                last_run=datetime.now(),
                last_error=str(e),
            )
            raise

    def _collect_all_physical_chunks_with_metadata(self) -> List[ChunkMetadata]:
        all_refs = list(self._chunk_enumerator.list_all_chunks())
        if self._metadata_provider is None:
            return [
                ChunkMetadata(
                    chunk_ref=ref,
                    size_bytes=0,
                )
                for ref in all_refs
            ]

        result = []
        for ref in all_refs:
            try:
                metadata = self._metadata_provider.get_metadata(ref)
                result.append(metadata)
            except ValueError:
                result.append(
                    ChunkMetadata(
                        chunk_ref=ref,
                        size_bytes=0,
                    )
                )
            except Exception as e:
                raise GarbageCollectionError(
                    f"Failed to get metadata for {ref}: {e}"
                ) from e
        return result

    def _execute_deletion_with_safety(
        self,
        candidates: tuple[DeletionCandidate, ...],
        force: bool
    ) -> tuple[set[ChunkRef], int]:
        """
        执行安全删除（Chunk 级别）。
        """
        if not candidates:
            return set(), 0

        # 预构建大小映射（O(N) 优化）
        size_map = {c.chunk_ref: c.size_bytes for c in candidates}

        ready_for_deletion: List[ChunkRef] = []
        reclaimed_total = 0

        for candidate in candidates:
            chunk_ref = candidate.chunk_ref

            # 安全检查：租约（如果启用）
            if not force and self._lease_manager is not None:
                try:
                    if not self._lease_manager.acquire(chunk_ref, self._lease_ttl_seconds, self._owner_id):
                        # CAS 竞争失败，跳过
                        continue
                except LeaseConflictError:
                    # 租约被其他节点持有且未过期，跳过
                    continue
                except LeaseAcquisitionError:
                    # 系统错误，终止本次 GC（安全优先）
                    raise

            # 标记删除（如果启用）
            if not force and self._deletion_marker_store is not None:
                try:
                    # 检查是否已标记
                    info = self._deletion_marker_store.get_deletion_info(chunk_ref)
                    if info is None:
                        self._deletion_marker_store.mark_for_deletion(chunk_ref, self._grace_period_seconds)
                    # 检查 grace period
                    self._deletion_marker_store.is_ready_for_physical_deletion(chunk_ref)
                except GracePeriodNotElapsedError:
                    # 未到时间，跳过
                    if not force and self._lease_manager is not None:
                        self._lease_manager.release(chunk_ref, self._owner_id)
                    continue
                except DeletionMarkerError:
                    # 标记失败，释放租约，跳过
                    if not force and self._lease_manager is not None:
                        self._lease_manager.release(chunk_ref, self._owner_id)
                    continue

            ready_for_deletion.append(chunk_ref)
            reclaimed_total += size_map.get(chunk_ref, 0)

        # 批量删除
        deleted = set()
        reclaimed = 0
        if ready_for_deletion:
            for i in range(0, len(ready_for_deletion), self._batch_size):
                batch = ready_for_deletion[i:i+self._batch_size]
                try:
                    self._delete_adapter.delete_chunks(batch)
                    deleted.update(batch)
                    reclaimed += sum(size_map.get(ref, 0) for ref in batch)
                    # 清除标记
                    if not force and self._deletion_marker_store is not None:
                        for ref in batch:
                            self._deletion_marker_store.clear_marker(ref)
                except Exception as e:
                    # 删除失败，抛出异常（租约在 finally 中释放）
                    raise DeletionFailedError(f"Batch deletion failed: {e} (failed batch size: {len(batch)})") from e
                finally:
                    # 释放租约（总是执行）
                    if not force and self._lease_manager is not None:
                        for ref in batch:
                            self._lease_manager.release(ref, self._owner_id)

        return deleted, reclaimed

    def stats(self) -> GCStats:
        """返回统计信息。"""
        return self._stats

    def reset_stats(self) -> None:
        """重置统计信息。"""
        self._stats = GCStats()