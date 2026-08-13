# src/writing/snapshot/runtime/remote/gc/retention_reconciler.py
"""
B4.10: RetentionReconciler — 负责保留策略一致性（protected marker 清理）
"""

import logging
from typing import Optional

from ...chunk_ref import ChunkRef
from .capability import ChunkExistenceChecker
from .deletion_marker_store import DeletionMarkerStore
from .scanner import DeletionMarkerScanner
from .retention_checker import ChunkRetentionChecker, RetentionDecision
from .models import MarkerReconciliationResult
from .errors import MarkerReconciliationError, DeletionMarkerError

logger = logging.getLogger(__name__)


class RetentionReconciler:
    """
    保留策略一致性协调器。

    职责：清理 protected markers（chunk 存在但应保留，marker 仍在）。

    依赖：
        - ChunkExistenceChecker: 确认 chunk 存在
        - ChunkRetentionChecker: 确认 chunk 应保留
    """

    def __init__(
        self,
        marker_scanner: DeletionMarkerScanner,
        marker_store: DeletionMarkerStore,
        existence_checker: ChunkExistenceChecker,
        retention_checker: ChunkRetentionChecker,
    ):
        self._marker_scanner = marker_scanner
        self._marker_store = marker_store
        self._existence_checker = existence_checker
        self._retention_checker = retention_checker

    def reconcile(self, auto_clear: bool = True) -> MarkerReconciliationResult:
        """
        执行保留策略一致性协调。

        Args:
            auto_clear: 是否自动清除 protected markers。
        """
        scanned = 0
        protected_found = 0
        protected_cleared = 0
        errors = 0

        for marker in self._marker_scanner.iter_pending_markers():
            scanned += 1

            try:
                chunk_exists = self._existence_checker.exists(marker)
            except Exception:
                raise

            if not chunk_exists:
                # 由 StorageReconciler 处理，此处跳过
                continue

            try:
                decision = self._retention_checker.decide(marker)
            except Exception:
                raise

            if decision == RetentionDecision.RETAIN:
                protected_found += 1
                if auto_clear:
                    try:
                        self._marker_store.clear_marker(marker)
                        protected_cleared += 1
                        logger.debug(f"Cleared protected marker: {marker}")
                    except DeletionMarkerError as e:
                        errors += 1
                        logger.warning(f"Failed to clear protected marker {marker}: {e}")

        return MarkerReconciliationResult(
            scanned_markers=scanned,
            protected_found=protected_found,
            protected_cleared=protected_cleared,
            errors=errors,
        )