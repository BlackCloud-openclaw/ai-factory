# src/writing/snapshot/runtime/remote/gc/models.py

from dataclasses import dataclass, field
from typing import FrozenSet, Optional, Tuple
from datetime import datetime

from ...id import SnapshotId
from ...chunk_ref import ChunkRef


@dataclass(frozen=True)
class ChunkMetadata:
    chunk_ref: ChunkRef
    size_bytes: int
    checksum: Optional[str] = None
    created_at: Optional[datetime] = None


@dataclass(frozen=True)
class ReachabilityGraph:
    reachable_snapshots: FrozenSet[SnapshotId] = field(default_factory=frozenset)
    reachable_chunks: FrozenSet[ChunkRef] = field(default_factory=frozenset)

    @property
    def snapshot_count(self) -> int:
        return len(self.reachable_snapshots)

    @property
    def chunk_count(self) -> int:
        return len(self.reachable_chunks)


@dataclass(frozen=True)
class DeletionCandidate:
    metadata: ChunkMetadata

    @property
    def chunk_ref(self) -> ChunkRef:
        return self.metadata.chunk_ref

    @property
    def size_bytes(self) -> int:
        return self.metadata.size_bytes


@dataclass(frozen=True)
class DeletionPlan:
    candidates: Tuple[DeletionCandidate, ...] = field(default_factory=tuple)

    @property
    def total_candidates(self) -> int:
        return len(self.candidates)

    @property
    def total_size_bytes(self) -> int:
        return sum(c.size_bytes for c in self.candidates)


@dataclass(frozen=True)
class GCResult:
    deleted_chunks: FrozenSet[ChunkRef] = field(default_factory=frozenset)
    reclaimed_bytes: int = 0
    dry_run: bool = False
    duration_ms: int = 0
    error: Optional[str] = None

    @property
    def deleted_count(self) -> int:
        return len(self.deleted_chunks)


@dataclass(frozen=True)
class GCStats:
    total_runs: int = 0
    total_deleted_chunks: int = 0
    total_reclaimed_bytes: int = 0
    total_duration_ms: int = 0
    last_run: Optional[datetime] = None
    last_error: Optional[str] = None


# ========== B4.8/B4.10 ==========

@dataclass(frozen=True)
class MarkerReconciliationResult:
    scanned_markers: int = 0
    stale_found: int = 0
    stale_cleared: int = 0
    protected_found: int = 0
    protected_cleared: int = 0
    errors: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "scanned_markers", "stale_found", "stale_cleared",
            "protected_found", "protected_cleared", "errors"
        ):
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative, got {value}")
        if self.stale_cleared > self.stale_found:
            raise ValueError(
                f"stale_cleared ({self.stale_cleared}) cannot exceed stale_found ({self.stale_found})"
            )
        if self.protected_cleared > self.protected_found:
            raise ValueError(
                f"protected_cleared ({self.protected_cleared}) cannot exceed protected_found ({self.protected_found})"
            )

    @property
    def issues_found(self) -> int:
        return self.stale_found + self.protected_found

    @property
    def issues_fixed(self) -> int:
        return self.stale_cleared + self.protected_cleared

    @property
    def issues_remaining(self) -> int:
        return (self.stale_found - self.stale_cleared) + (self.protected_found - self.protected_cleared) + self.errors

    def __str__(self) -> str:
        return (
            f"MarkerReconciliationResult("
            f"scanned={self.scanned_markers}, "
            f"stale_found={self.stale_found}, "
            f"stale_cleared={self.stale_cleared}, "
            f"protected_found={self.protected_found}, "
            f"protected_cleared={self.protected_cleared}, "
            f"errors={self.errors})"
        )