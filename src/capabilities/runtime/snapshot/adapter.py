# src/capabilities/runtime/snapshot/adapter.py
"""
Phase 11.2.3: Snapshot Runtime Capability Adapters
"""

from typing import Optional

from src.writing.snapshot.runtime.incremental import ChunkRepository, VersionStore, ChunkRepository
from src.writing.snapshot.runtime.incremental.incremental_transport import IncrementalTransport
from src.writing.snapshot.runtime.serializers import SerializerResolver
from src.writing.snapshot.runtime.compression import CompressionResolver
from src.writing.snapshot.runtime.chunking import ChunkingStrategy, FixedChunkStrategy
from src.writing.snapshot.runtime.remote import RemoteChunkRepository
from src.writing.snapshot.runtime.remote.s3 import S3Client, S3Config

from .implementation import (
    SnapshotRepositoryCapability,
    SnapshotVersionStoreCapability,
    SnapshotTransportCapability,
)


class SnapshotCapabilityAdapter:
    """
    Factory for Snapshot Runtime Capabilities.
    """

    @staticmethod
    def create_repository(
        chunk_repository: ChunkRepository,
    ) -> SnapshotRepositoryCapability:
        """创建 ChunkRepository Capability。"""
        return SnapshotRepositoryCapability(chunk_repository)

    @staticmethod
    def create_version_store(
        version_store: VersionStore,
    ) -> SnapshotVersionStoreCapability:
        """创建 VersionStore Capability。"""
        return SnapshotVersionStoreCapability(version_store)

    @staticmethod
    def create_transport(
        transport: IncrementalTransport,
    ) -> SnapshotTransportCapability:
        """创建 IncrementalTransport Capability。"""
        return SnapshotTransportCapability(transport)

    @staticmethod
    def create_default_transport(
        repository: ChunkRepository,
        version_store: VersionStore,
        *,
        serializer_resolver: Optional[SerializerResolver] = None,
        compression_resolver: Optional[CompressionResolver] = None,
        strategy: Optional[ChunkingStrategy] = None,
        default_serializer_id: str = "builtin.json",
        default_codec_id: str = "builtin.identity",
        max_chain_depth: int = 32,
    ) -> SnapshotTransportCapability:
        """创建默认的 IncrementalTransport Capability。"""
        from src.writing.snapshot.runtime.incremental.incremental_transport import IncrementalTransport
        from src.writing.snapshot.runtime.serializers.registry import SerializerRegistry
        from src.writing.snapshot.runtime.compression.registry import CompressionRegistry

        if serializer_resolver is None:
            serializer_resolver = SerializerRegistry.with_builtin()
        if compression_resolver is None:
            compression_resolver = CompressionRegistry.with_builtin()
        if strategy is None:
            strategy = FixedChunkStrategy(1024 * 1024)

        transport = IncrementalTransport(
            repository=repository,
            serializer_resolver=serializer_resolver,
            compression_resolver=compression_resolver,
            strategy=strategy,
            default_serializer_id=default_serializer_id,
            default_codec_id=default_codec_id,
            max_chain_depth=max_chain_depth,
        )

        return SnapshotTransportCapability(transport)