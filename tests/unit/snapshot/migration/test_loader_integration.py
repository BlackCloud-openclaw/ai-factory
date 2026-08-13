# tests/unit/snapshot/migration/test_loader_integration.py

import pytest

from src.writing.snapshot.migration import (
    CurrentSchemaProvider,
    DefaultMigrationContextFactory,
    MigrationEdge,
    MigrationGraph,
    MigrationRegistry,
    MigrationRuntime,
    PathStrategy,
    RawSnapshot,
    SchemaVersion,
    SnapshotMigrator,
    SnapshotVersionTooNewError,
    StaticSchemaProvider,
    VersionNode,
    VersionType,
    MigrationPathNotFoundError,
)


# ================================================================
# 测试 Fixtures
# ================================================================

def build_migration_graph() -> MigrationGraph:
    """构建 1.0 -> 1.1 -> 2.0 的迁移图。"""
    registry = MigrationRegistry()

    nodes = [
        VersionNode(version=SchemaVersion(1, 0), version_type=VersionType.MINOR),
        VersionNode(version=SchemaVersion(1, 1), version_type=VersionType.MINOR),
        VersionNode(version=SchemaVersion(2, 0), version_type=VersionType.MINOR),
    ]
    for node in nodes:
        registry.register_node(node)

    def upcaster_1_0_to_1_1(snap, ctx):
        data = dict(snap._data)
        return RawSnapshot.from_mapping(schema_version=SchemaVersion(1, 1), data=data)

    def upcaster_1_1_to_2_0(snap, ctx):
        data = dict(snap._data)
        return RawSnapshot.from_mapping(schema_version=SchemaVersion(2, 0), data=data)

    registry.register_edge(MigrationEdge(
        from_version=SchemaVersion(1, 0),
        to_version=SchemaVersion(1, 1),
        upcaster=upcaster_1_0_to_1_1,
    ))
    registry.register_edge(MigrationEdge(
        from_version=SchemaVersion(1, 1),
        to_version=SchemaVersion(2, 0),
        upcaster=upcaster_1_1_to_2_0,
    ))

    return registry.build()


@pytest.fixture
def migrator() -> SnapshotMigrator:
    graph = build_migration_graph()
    return MigrationRuntime(graph, strategy=PathStrategy.MINOR_FIRST)


@pytest.fixture
def schema_provider_v2() -> CurrentSchemaProvider:
    return StaticSchemaProvider(SchemaVersion(2, 0))


@pytest.fixture
def context_factory() -> DefaultMigrationContextFactory:
    return DefaultMigrationContextFactory()


@pytest.fixture
def snapshot_v1_0() -> RawSnapshot:
    data = {"title": "Test", "chapters": 10}
    return RawSnapshot.from_mapping(
        schema_version=SchemaVersion(1, 0),
        data=data,
    )


# ================================================================
# 模拟 Loader
# ================================================================

class MockSnapshotLoader:
    """模拟 Loader，用于测试迁移逻辑，避免文件 I/O。"""

    def __init__(
        self,
        migrator: SnapshotMigrator,
        schema_provider: CurrentSchemaProvider,
        context_factory: DefaultMigrationContextFactory,
    ):
        self._migrator = migrator
        self._schema_provider = schema_provider
        self._context_factory = context_factory

    def load(self, snapshot: RawSnapshot) -> RawSnapshot:
        target = self._schema_provider.get()

        if snapshot.schema_version == target:
            return snapshot

        if snapshot.schema_version.is_newer_than(target):
            raise SnapshotVersionTooNewError(
                f"Snapshot version {snapshot.schema_version} is newer than "
                f"current supported version {target}"
            )

        context = self._context_factory.create()
        return self._migrator.migrate(snapshot, target, context)


# ================================================================
# 集成测试
# ================================================================

class TestLoaderIntegration:
    def test_migrate_v1_0_to_v2_0(self, migrator, schema_provider_v2, context_factory, snapshot_v1_0):
        loader = MockSnapshotLoader(migrator, schema_provider_v2, context_factory)
        result = loader.load(snapshot_v1_0)
        assert result.schema_version == SchemaVersion(2, 0)
        assert result.get("title") == "Test"
        assert result.get("chapters") == 10

    def test_v2_0_no_migration(self, migrator, schema_provider_v2, context_factory):
        snapshot = RawSnapshot.from_mapping(
            schema_version=SchemaVersion(2, 0),
            data={"title": "Already Latest"},
        )
        loader = MockSnapshotLoader(migrator, schema_provider_v2, context_factory)
        result = loader.load(snapshot)
        assert result is snapshot  # 返回原对象
        assert result.schema_version == SchemaVersion(2, 0)

    def test_snapshot_newer_than_current_raises(self, migrator, schema_provider_v2, context_factory):
        snapshot = RawSnapshot.from_mapping(
            schema_version=SchemaVersion(3, 0),
            data={"title": "Future Version"},
        )
        loader = MockSnapshotLoader(migrator, schema_provider_v2, context_factory)
        with pytest.raises(SnapshotVersionTooNewError, match="newer than"):
            loader.load(snapshot)

    def test_no_migration_path_raises(self, migrator, schema_provider_v2, context_factory):
        # 目标版本 3.0，但图中只有到 2.0 的路径
        schema_provider_v3 = StaticSchemaProvider(SchemaVersion(3, 0))
        loader = MockSnapshotLoader(migrator, schema_provider_v3, context_factory)
        snapshot = RawSnapshot.from_mapping(
            schema_version=SchemaVersion(1, 0),
            data={"title": "Test"},
        )
        with pytest.raises(MigrationPathNotFoundError, match="No migration path"):
            loader.load(snapshot)