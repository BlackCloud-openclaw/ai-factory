# src/capabilities/runtime/snapshot/constants.py
"""
Phase 11.2.3: Snapshot Runtime Capability Constants
"""

from packaging.version import Version

from src.capabilities.spec import CapabilitySpec, CapabilityMetadata

# ========== ChunkRepository ==========
SNAPSHOT_REPOSITORY_ID = "builtin.runtime.snapshot.repository"
SNAPSHOT_REPOSITORY_VERSION = "1.0"

SNAPSHOT_REPOSITORY_SPEC = CapabilitySpec(
    id=SNAPSHOT_REPOSITORY_ID,
    version=Version(SNAPSHOT_REPOSITORY_VERSION),
    metadata=CapabilityMetadata(
        display_name="Snapshot Chunk Repository",
        description="ChunkRepository for snapshot storage (S3/File/Memory)",
        tags=("snapshot", "storage", "runtime"),
    ),
)

# ========== VersionStore ==========
SNAPSHOT_VERSION_STORE_ID = "builtin.runtime.snapshot.version_store"
SNAPSHOT_VERSION_STORE_VERSION = "1.0"

SNAPSHOT_VERSION_STORE_SPEC = CapabilitySpec(
    id=SNAPSHOT_VERSION_STORE_ID,
    version=Version(SNAPSHOT_VERSION_STORE_VERSION),
    metadata=CapabilityMetadata(
        display_name="Snapshot Version Store",
        description="VersionStore for snapshot version metadata",
        tags=("snapshot", "version", "runtime"),
    ),
)

# ========== IncrementalTransport ==========
# 注意：Transport 作为 Runtime Service Capability，但只由 SnapshotRuntimeService 消费
SNAPSHOT_TRANSPORT_ID = "builtin.runtime.snapshot.transport"
SNAPSHOT_TRANSPORT_VERSION = "1.0"

SNAPSHOT_TRANSPORT_SPEC = CapabilitySpec(
    id=SNAPSHOT_TRANSPORT_ID,
    version=Version(SNAPSHOT_TRANSPORT_VERSION),
    metadata=CapabilityMetadata(
        display_name="Snapshot Incremental Transport",
        description="IncrementalTransport for snapshot I/O (Runtime Service Level)",
        tags=("snapshot", "transport", "runtime"),
    ),
)