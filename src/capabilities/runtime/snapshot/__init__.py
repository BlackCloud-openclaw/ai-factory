# src/capabilities/runtime/snapshot/__init__.py
"""
Phase 11.2.3: Snapshot Runtime Capability Module
"""

from .constants import (
    SNAPSHOT_REPOSITORY_ID,
    SNAPSHOT_REPOSITORY_VERSION,
    SNAPSHOT_REPOSITORY_SPEC,
    SNAPSHOT_VERSION_STORE_ID,
    SNAPSHOT_VERSION_STORE_VERSION,
    SNAPSHOT_VERSION_STORE_SPEC,
    SNAPSHOT_TRANSPORT_ID,
    SNAPSHOT_TRANSPORT_VERSION,
    SNAPSHOT_TRANSPORT_SPEC,
)
from .implementation import (
    SnapshotRepositoryCapability,
    SnapshotVersionStoreCapability,
    SnapshotTransportCapability,
)
from .adapter import SnapshotCapabilityAdapter

__all__ = [
    "SNAPSHOT_REPOSITORY_ID",
    "SNAPSHOT_REPOSITORY_VERSION",
    "SNAPSHOT_REPOSITORY_SPEC",
    "SNAPSHOT_VERSION_STORE_ID",
    "SNAPSHOT_VERSION_STORE_VERSION",
    "SNAPSHOT_VERSION_STORE_SPEC",
    "SNAPSHOT_TRANSPORT_ID",
    "SNAPSHOT_TRANSPORT_VERSION",
    "SNAPSHOT_TRANSPORT_SPEC",
    "SnapshotRepositoryCapability",
    "SnapshotVersionStoreCapability",
    "SnapshotTransportCapability",
    "SnapshotCapabilityAdapter",
]