# src/capabilities/runtime/snapshot/implementation.py
"""
Phase 11.2.3: Snapshot Runtime Capability Implementations
"""

from src.writing.snapshot.runtime.incremental import ChunkRepository, VersionStore
from src.writing.snapshot.runtime.incremental.incremental_transport import IncrementalTransport


class SnapshotRepositoryCapability:
    """ChunkRepository Runtime Capability."""

    def __init__(self, repository: ChunkRepository):
        self._repository = repository

    def get(self) -> ChunkRepository:
        return self._repository


class SnapshotVersionStoreCapability:
    """VersionStore Runtime Capability."""

    def __init__(self, version_store: VersionStore):
        self._version_store = version_store

    def get(self) -> VersionStore:
        return self._version_store


class SnapshotTransportCapability:
    """
    IncrementalTransport Runtime Capability.

    注意：此 Capability 不直接暴露给业务层。
    仅由 SnapshotRuntimeService 或 Composition Root 内部使用。
    """

    def __init__(self, transport: IncrementalTransport):
        self._transport = transport

    def get(self) -> IncrementalTransport:
        return self._transport