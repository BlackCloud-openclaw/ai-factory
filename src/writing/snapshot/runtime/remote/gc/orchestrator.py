# src/writing/snapshot/runtime/remote/gc/orchestrator.py
"""
B4.11: GCOrchestrator — GC 工作流编排器（依赖注入版）
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional

from .collector import GarbageCollector
from .storage_reconciler import StorageReconciler
from .retention_reconciler import RetentionReconciler
from .lease import LeaseManager
from .models import GCResult, MarkerReconciliationResult
from .errors import GarbageCollectionError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrchestratorConfig:
    enable_storage_reconcile: bool = True
    enable_retention_reconcile: bool = True
    auto_clear_reconcile: bool = True
    dry_run: bool = False
    scope_ttl_seconds: int = 600
    owner_id: Optional[str] = None


@dataclass
class OrchestratorResult:
    gc_result: Optional[GCResult] = None
    storage_reconcile_result: Optional[MarkerReconciliationResult] = None
    retention_reconcile_result: Optional[MarkerReconciliationResult] = None
    total_duration_ms: int = 0
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    @property
    def deleted_count(self) -> int:
        return self.gc_result.deleted_count if self.gc_result else 0

    @property
    def reclaimed_bytes(self) -> int:
        return self.gc_result.reclaimed_bytes if self.gc_result else 0


class GCOrchestrator:
    """
    GC 工作流编排器（依赖注入）。

    只负责调度顺序，不负责构建任何子组件。
    所有组件由外部（Composition Root）注入。
    """

    def __init__(
        self,
        storage_reconciler: StorageReconciler,
        retention_reconciler: RetentionReconciler,
        garbage_collector: GarbageCollector,
        lease_manager: LeaseManager,
        config: Optional[OrchestratorConfig] = None,
    ):
        self._storage_reconciler = storage_reconciler
        self._retention_reconciler = retention_reconciler
        self._garbage_collector = garbage_collector
        self._lease_manager = lease_manager
        self._config = config or OrchestratorConfig()

    def run(self) -> OrchestratorResult:
        """执行完整 GC 工作流。"""
        start_time = time.perf_counter()
        result = OrchestratorResult()
        scope_id = self._config.owner_id or f"gc-session-{int(start_time)}"

        try:
            # Step 1: 获取作用域租约（上下文管理器）
            with self._lease_manager.scope(scope_id, self._config.scope_ttl_seconds):
                logger.info(f"GC session {scope_id}: scope lease acquired")

                # Step 2: Storage Reconciliation
                if self._config.enable_storage_reconcile:
                    logger.info(f"GC session {scope_id}: storage reconciliation...")
                    result.storage_reconcile_result = self._storage_reconciler.reconcile(
                        auto_clear=self._config.auto_clear_reconcile
                    )
                    logger.debug(f"Storage reconcile: {result.storage_reconcile_result}")

                # Step 3: Retention Reconciliation
                if self._config.enable_retention_reconcile:
                    logger.info(f"GC session {scope_id}: retention reconciliation...")
                    result.retention_reconcile_result = self._retention_reconciler.reconcile(
                        auto_clear=self._config.auto_clear_reconcile
                    )
                    logger.debug(f"Retention reconcile: {result.retention_reconcile_result}")

                # Step 4: Execute GC（Collector 内部包含 Analysis + Planning + Deletion）
                logger.info(f"GC session {scope_id}: executing garbage collection...")
                result.gc_result = self._garbage_collector.collect(dry_run=self._config.dry_run)
                logger.info(
                    f"GC session {scope_id}: deletion complete, "
                    f"{result.gc_result.deleted_count} chunks deleted, "
                    f"{result.gc_result.reclaimed_bytes} bytes reclaimed"
                )

        except Exception as e:
            result.error = str(e)
            logger.error(f"GC session {scope_id} failed: {e}", exc_info=True)
            raise

        result.total_duration_ms = int((time.perf_counter() - start_time) * 1000)
        logger.info(f"GC session {scope_id} completed in {result.total_duration_ms}ms")
        return result

    def run_reconciliation(self) -> tuple[Optional[MarkerReconciliationResult], Optional[MarkerReconciliationResult]]:
        """
        仅执行 Reconciliation（不执行删除）。
        用于定期维护或监控。
        """
        storage_result = None
        retention_result = None

        if self._config.enable_storage_reconcile:
            storage_result = self._storage_reconciler.reconcile(
                auto_clear=self._config.auto_clear_reconcile
            )

        if self._config.enable_retention_reconcile:
            retention_result = self._retention_reconciler.reconcile(
                auto_clear=self._config.auto_clear_reconcile
            )

        return storage_result, retention_result