# src/writing/bootstrap/runtime_capabilities.py
"""
Phase 11.2.2/11.2.3: Runtime Capabilities Composition Root
"""

from src.capabilities.runtime import RuntimeCapabilityRegistry, FrozenRuntimeCapabilityRegistry
from src.capabilities.audit import (
    AUDIT_COORDINATOR_SPEC,
    AuditCapabilityAdapter,
)
from src.capabilities.runtime.snapshot import (
    SNAPSHOT_REPOSITORY_SPEC,
    SNAPSHOT_VERSION_STORE_SPEC,
    SNAPSHOT_TRANSPORT_SPEC,
    SnapshotCapabilityAdapter,
)
from src.writing.snapshot.runtime.incremental import MemoryChunkRepository, MemoryVersionStore


def build_runtime_capabilities() -> FrozenRuntimeCapabilityRegistry:
    """
    构建 Runtime Capability Registry（已冻结）。
    """
    registry = RuntimeCapabilityRegistry()

    # ----- 1. Audit Capabilities -----
    audit_capability = AuditCapabilityAdapter.create()
    registry.register(AUDIT_COORDINATOR_SPEC, audit_capability)

    # ----- 2. Snapshot Capabilities -----
    # 2.1 ChunkRepository（默认使用 Memory 实现，可替换为 File/S3）
    repository = MemoryChunkRepository()
    repo_capability = SnapshotCapabilityAdapter.create_repository(repository)
    registry.register(SNAPSHOT_REPOSITORY_SPEC, repo_capability)

    # 2.2 VersionStore（默认使用 Memory 实现，可替换为 File/S3）
    version_store = MemoryVersionStore()
    version_capability = SnapshotCapabilityAdapter.create_version_store(version_store)
    registry.register(SNAPSHOT_VERSION_STORE_SPEC, version_capability)

    # 2.3 IncrementalTransport（使用默认配置）
    transport_capability = SnapshotCapabilityAdapter.create_default_transport(
        repository=repository,
        version_store=version_store,
    )
    registry.register(SNAPSHOT_TRANSPORT_SPEC, transport_capability)

    # ----- 3. 未来扩展 -----
    # registry.register(PAYLOAD_RESOLVER_SPEC, payload_capability)

    return registry.freeze()