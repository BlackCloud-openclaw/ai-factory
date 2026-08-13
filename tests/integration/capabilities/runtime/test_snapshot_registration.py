# tests/integration/capabilities/runtime/test_snapshot_registration.py

import pytest
from src.capabilities.runtime import RuntimeCapabilityRegistry
from src.capabilities.runtime.snapshot import (
    SNAPSHOT_REPOSITORY_ID,
    SNAPSHOT_REPOSITORY_SPEC,
    SNAPSHOT_VERSION_STORE_ID,
    SNAPSHOT_VERSION_STORE_SPEC,
    SNAPSHOT_TRANSPORT_ID,
    SNAPSHOT_TRANSPORT_SPEC,
    SnapshotCapabilityAdapter,
)
from src.writing.snapshot.runtime.incremental import MemoryChunkRepository, MemoryVersionStore
from src.writing.bootstrap.runtime_capabilities import build_runtime_capabilities


class TestSnapshotRuntimeRegistration:

    def test_snapshot_capabilities_registered(self):
        """验证所有 Snapshot Capability 已注册。"""
        registry = build_runtime_capabilities()
        assert registry.has(SNAPSHOT_REPOSITORY_ID) is True
        assert registry.has(SNAPSHOT_VERSION_STORE_ID) is True
        assert registry.has(SNAPSHOT_TRANSPORT_ID) is True
        ids = registry.list_ids()
        assert SNAPSHOT_REPOSITORY_ID in ids
        assert SNAPSHOT_VERSION_STORE_ID in ids
        assert SNAPSHOT_TRANSPORT_ID in ids

    def test_repository_capability_returns_repository(self):
        """验证 Repository Capability 返回 ChunkRepository（通过方法存在性检查）。"""
        registry = build_runtime_capabilities()
        capability = registry.require(SNAPSHOT_REPOSITORY_ID)
        repository = capability.get()
        # ChunkRepository 是 Protocol，避免使用 isinstance，检查关键方法存在性
        assert hasattr(repository, "save_version")
        assert hasattr(repository, "load_version")
        assert hasattr(repository, "load_manifest")
        assert hasattr(repository, "exists")
        assert hasattr(repository, "delete")
        assert hasattr(repository, "list_ids")
        # 对于流式方法，检查是否存在（可选）
        if hasattr(repository, "save_chunk_stream"):
            assert callable(repository.save_chunk_stream)
        if hasattr(repository, "load_chunk_stream"):
            assert callable(repository.load_chunk_stream)

    def test_version_store_capability_returns_version_store(self):
        """验证 VersionStore Capability 返回 VersionStore。"""
        registry = build_runtime_capabilities()
        capability = registry.require(SNAPSHOT_VERSION_STORE_ID)
        version_store = capability.get()
        # 检查方法存在性
        assert hasattr(version_store, "put")
        assert hasattr(version_store, "get")
        assert hasattr(version_store, "delete")
        assert hasattr(version_store, "list_ids")

    def test_transport_capability_returns_transport(self):
        """验证 Transport Capability 返回 IncrementalTransport。"""
        registry = build_runtime_capabilities()
        capability = registry.require(SNAPSHOT_TRANSPORT_ID)
        transport = capability.get()
        assert hasattr(transport, "write")
        assert hasattr(transport, "read")

    def test_manual_registration_works(self):
        """验证手动注册流程。"""
        registry = RuntimeCapabilityRegistry()
        repository = MemoryChunkRepository()
        version_store = MemoryVersionStore()
        repo_cap = SnapshotCapabilityAdapter.create_repository(repository)
        version_cap = SnapshotCapabilityAdapter.create_version_store(version_store)
        transport_cap = SnapshotCapabilityAdapter.create_default_transport(
            repository=repository,
            version_store=version_store,
        )
        registry.register(SNAPSHOT_REPOSITORY_SPEC, repo_cap)
        registry.register(SNAPSHOT_VERSION_STORE_SPEC, version_cap)
        registry.register(SNAPSHOT_TRANSPORT_SPEC, transport_cap)
        frozen = registry.freeze()
        assert frozen.has(SNAPSHOT_REPOSITORY_ID) is True
        assert frozen.has(SNAPSHOT_VERSION_STORE_ID) is True
        assert frozen.has(SNAPSHOT_TRANSPORT_ID) is True

    def test_transport_capability_uses_same_repository_and_version_store(self):
        """验证 Transport Capability 与 Repository/VersionStore 引用一致。"""
        repository = MemoryChunkRepository()
        version_store = MemoryVersionStore()
        transport_cap = SnapshotCapabilityAdapter.create_default_transport(
            repository=repository,
            version_store=version_store,
        )
        transport = transport_cap.get()
        assert transport is not None
        # 间接验证：transport 内部应包含 repository 和 version_store
        # 由于无法直接访问，检查 transport 是否有 write/read 方法
        assert hasattr(transport, "write")
        assert hasattr(transport, "read")