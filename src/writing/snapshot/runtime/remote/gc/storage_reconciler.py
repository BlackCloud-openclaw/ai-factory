# src/writing/snapshot/runtime/remote/gc/storage_reconciler.py
"""
B4.10: StorageReconciler — 仅负责存储一致性（stale marker 清理）
"""

import logging
from typing import Optional

from ...chunk_ref import ChunkRef
from .capability import ChunkExistenceChecker, ChunkEnumerator
from .deletion_marker_store import DeletionMarkerStore
from .scanner import DeletionMarkerScanner
from .existence import EnumeratorExistenceChecker
from .models import MarkerReconciliationResult
from .errors import MarkerReconciliationError, DeletionMarkerError

logger = logging.getLogger(__name__)


class StorageReconciler:
    """
    存储一致性协调器。

    职责：清理 stale markers（chunk 已不存在但 marker 仍在）。

    统一依赖 ChunkExistenceChecker，不感知底层实现。
    """

    def __init__(
        self,
        marker_scanner: DeletionMarkerScanner,
        marker_store: DeletionMarkerStore,
        existence_checker: ChunkExistenceChecker,
    ):
        self._marker_scanner = marker_scanner
        self._marker_store = marker_store
        self._existence_checker = existence_checker

    def reconcile(self, auto_clear: bool = True) -> MarkerReconciliationResult:
        """
        执行存储一致性协调。

        Args:
            auto_clear: 是否自动清除 stale markers。
        """
        scanned = 0
        stale_found = 0
        stale_cleared = 0
        errors = 0

        for marker in self._marker_scanner.iter_pending_markers():
            scanned += 1

            try:
                chunk_exists = self._existence_checker.exists(marker)
            except Exception:
                raise

            if not chunk_exists:
                stale_found += 1
                if auto_clear:
                    try:
                        self._marker_store.clear_marker(marker)
                        stale_cleared += 1
                        logger.debug(f"Cleared stale marker: {marker}")
                    except DeletionMarkerError as e:
                        errors += 1
                        logger.warning(f"Failed to clear stale marker {marker}: {e}")

        return MarkerReconciliationResult(
            scanned_markers=scanned,
            stale_found=stale_found,
            stale_cleared=stale_cleared,
            errors=errors,
        )