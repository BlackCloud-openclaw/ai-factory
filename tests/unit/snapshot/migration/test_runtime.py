# tests/unit/snapshot/migration/test_runtime.py

import pytest

from src.writing.snapshot.migration import (
    MigrationContext,
    MigrationEdge,
    MigrationGraph,
    MigrationRuntime,
    PathStrategy,
    RawSnapshot,
    SchemaVersion,
    VersionNode,
    VersionType,
    MigrationPathNotFoundError,
    MigrationExecutionError,
)


# ================================================================
# Fixtures
# ================================================================

@pytest.fixture
def simple_graph() -> MigrationGraph:
    """图：1.0 -> 1.1 -> 2.0"""
    graph = MigrationGraph()
    for v in [SchemaVersion(1, 0), SchemaVersion(1, 1), SchemaVersion(2, 0)]:
        graph.add_node(VersionNode(version=v, version_type=VersionType.MINOR))

    def upcaster_1_0_to_1_1(snap, ctx):
        data = dict(snap._data)
        return RawSnapshot.from_mapping(
            schema_version=SchemaVersion(1, 1),
            data=data,
        )

    def upcaster_1_1_to_2_0(snap, ctx):
        data = dict(snap._data)
        return RawSnapshot.from_mapping(
            schema_version=SchemaVersion(2, 0),
            data=data,
        )

    graph.add_edge(MigrationEdge(
        from_version=SchemaVersion(1, 0),
        to_version=SchemaVersion(1, 1),
        upcaster=upcaster_1_0_to_1_1,
    ))
    graph.add_edge(MigrationEdge(
        from_version=SchemaVersion(1, 1),
        to_version=SchemaVersion(2, 0),
        upcaster=upcaster_1_1_to_2_0,
    ))
    return graph


@pytest.fixture
def snapshot_v1_0() -> RawSnapshot:
    data = {"title": "Test", "chapters": 10}
    return RawSnapshot.from_mapping(
        schema_version=SchemaVersion(1, 0),
        data=data,
    )


@pytest.fixture
def migration_context() -> MigrationContext:
    return MigrationContext()


# ================================================================
# 基础迁移测试
# ================================================================

class TestMigrationRuntime:
    def test_migrate_success(self, simple_graph, snapshot_v1_0, migration_context):
        runtime = MigrationRuntime(simple_graph)
        result = runtime.migrate(
            snapshot_v1_0,
            target_version=SchemaVersion(2, 0),
            context=migration_context,
        )
        assert result.schema_version == SchemaVersion(2, 0)
        assert result.get("title") == "Test"
        assert result.get("chapters") == 10

    def test_migrate_source_equals_target(self, simple_graph, snapshot_v1_0, migration_context):
        runtime = MigrationRuntime(simple_graph)
        result = runtime.migrate(
            snapshot_v1_0,
            target_version=SchemaVersion(1, 0),
            context=migration_context,
        )
        assert result is snapshot_v1_0

    def test_migrate_no_path(self, simple_graph, snapshot_v1_0, migration_context):
        runtime = MigrationRuntime(simple_graph)
        with pytest.raises(MigrationPathNotFoundError, match="No migration path"):
            runtime.migrate(
                snapshot_v1_0,
                target_version=SchemaVersion(3, 0),
                context=migration_context,
            )

    def test_migrate_with_shortest_strategy(self, simple_graph, snapshot_v1_0, migration_context):
        runtime = MigrationRuntime(simple_graph, strategy=PathStrategy.SHORTEST)
        result = runtime.migrate(
            snapshot_v1_0,
            target_version=SchemaVersion(2, 0),
            context=migration_context,
        )
        assert result.schema_version == SchemaVersion(2, 0)


# ================================================================
# 错误边界测试
# ================================================================

class TestErrorBoundaries:
    def test_upcaster_raises_exception_wrapped(self, snapshot_v1_0, migration_context):
        graph = MigrationGraph()
        for v in [SchemaVersion(1, 0), SchemaVersion(1, 1), SchemaVersion(2, 0)]:
            graph.add_node(VersionNode(version=v, version_type=VersionType.MINOR))

        def failing_upcaster(snap, ctx):
            raise ValueError("Something went wrong")

        graph.add_edge(MigrationEdge(
            from_version=SchemaVersion(1, 0),
            to_version=SchemaVersion(1, 1),
            upcaster=failing_upcaster,
        ))

        def dummy_upcaster(snap, ctx):
            data = dict(snap._data)
            return RawSnapshot.from_mapping(schema_version=SchemaVersion(2, 0), data=data)
        graph.add_edge(MigrationEdge(
            from_version=SchemaVersion(1, 1),
            to_version=SchemaVersion(2, 0),
            upcaster=dummy_upcaster,
        ))

        runtime = MigrationRuntime(graph)
        with pytest.raises(MigrationExecutionError, match="Migration failed at 1.0 → 1.1"):
            runtime.migrate(snapshot_v1_0, SchemaVersion(2, 0), migration_context)

    def test_upcaster_returns_non_rawsnapshot(self, snapshot_v1_0, migration_context):
        graph = MigrationGraph()
        for v in [SchemaVersion(1, 0), SchemaVersion(1, 1), SchemaVersion(2, 0)]:
            graph.add_node(VersionNode(version=v, version_type=VersionType.MINOR))

        graph.add_edge(MigrationEdge(
            from_version=SchemaVersion(1, 0),
            to_version=SchemaVersion(1, 1),
            upcaster=lambda snap, ctx: {"invalid": "dict"},
        ))

        def dummy_upcaster(snap, ctx):
            data = dict(snap._data)
            return RawSnapshot.from_mapping(schema_version=SchemaVersion(2, 0), data=data)
        graph.add_edge(MigrationEdge(
            from_version=SchemaVersion(1, 1),
            to_version=SchemaVersion(2, 0),
            upcaster=dummy_upcaster,
        ))

        runtime = MigrationRuntime(graph)
        with pytest.raises(MigrationExecutionError, match="Upcaster must return RawSnapshot"):
            runtime.migrate(snapshot_v1_0, SchemaVersion(2, 0), migration_context)

    def test_final_version_mismatch(self, snapshot_v1_0, migration_context):
        graph = MigrationGraph()
        for v in [SchemaVersion(1, 0), SchemaVersion(1, 1), SchemaVersion(2, 0)]:
            graph.add_node(VersionNode(version=v, version_type=VersionType.MINOR))

        def upcaster_1_0_to_1_1(snap, ctx):
            data = dict(snap._data)
            return RawSnapshot.from_mapping(schema_version=SchemaVersion(1, 1), data=data)
        graph.add_edge(MigrationEdge(
            from_version=SchemaVersion(1, 0),
            to_version=SchemaVersion(1, 1),
            upcaster=upcaster_1_0_to_1_1,
        ))

        def wrong_upcaster(snap, ctx):
            data = dict(snap._data)
            return RawSnapshot.from_mapping(schema_version=SchemaVersion(1, 1), data=data)
        graph.add_edge(MigrationEdge(
            from_version=SchemaVersion(1, 1),
            to_version=SchemaVersion(2, 0),
            upcaster=wrong_upcaster,
        ))

        runtime = MigrationRuntime(graph)
        with pytest.raises(MigrationExecutionError, match="incorrect version"):
            runtime.migrate(snapshot_v1_0, SchemaVersion(2, 0), migration_context)

    def test_input_type_error_snapshot(self, simple_graph, migration_context):
        runtime = MigrationRuntime(simple_graph)
        with pytest.raises(TypeError, match="snapshot must be RawSnapshot"):
            runtime.migrate(
                snapshot={"not": "raw"},  # type: ignore
                target_version=SchemaVersion(2, 0),
                context=migration_context,
            )

    def test_input_type_error_context(self, simple_graph, snapshot_v1_0):
        runtime = MigrationRuntime(simple_graph)
        with pytest.raises(TypeError, match="context must be MigrationContext"):
            runtime.migrate(
                snapshot=snapshot_v1_0,
                target_version=SchemaVersion(2, 0),
                context={},  # type: ignore
            )


# ================================================================
# 执行顺序测试
# ================================================================

class TestExecutionOrder:
    def test_upcaster_execution_order(self, migration_context):
        graph = MigrationGraph()
        for v in [SchemaVersion(1, 0), SchemaVersion(1, 1), SchemaVersion(2, 0), SchemaVersion(3, 0)]:
            graph.add_node(VersionNode(version=v, version_type=VersionType.MINOR))

        execution_log = []

        def make_upcaster(edge_name, from_ver, to_ver):
            def upcaster(snap, ctx):
                execution_log.append((edge_name, from_ver, to_ver))
                data = dict(snap._data) if hasattr(snap, "_data") else {}
                return RawSnapshot.from_mapping(
                    schema_version=to_ver,
                    data=data,
                )
            return upcaster

        graph.add_edge(MigrationEdge(
            from_version=SchemaVersion(1, 0),
            to_version=SchemaVersion(1, 1),
            upcaster=make_upcaster("A", SchemaVersion(1, 0), SchemaVersion(1, 1)),
        ))
        graph.add_edge(MigrationEdge(
            from_version=SchemaVersion(1, 1),
            to_version=SchemaVersion(2, 0),
            upcaster=make_upcaster("B", SchemaVersion(1, 1), SchemaVersion(2, 0)),
        ))
        graph.add_edge(MigrationEdge(
            from_version=SchemaVersion(2, 0),
            to_version=SchemaVersion(3, 0),
            upcaster=make_upcaster("C", SchemaVersion(2, 0), SchemaVersion(3, 0)),
        ))

        snapshot = RawSnapshot.from_mapping(
            schema_version=SchemaVersion(1, 0),
            data={"test": 1},
        )
        runtime = MigrationRuntime(graph)
        result = runtime.migrate(
            snapshot,
            target_version=SchemaVersion(3, 0),
            context=migration_context,
        )
        assert result.schema_version == SchemaVersion(3, 0)
        expected = [
            ("A", SchemaVersion(1, 0), SchemaVersion(1, 1)),
            ("B", SchemaVersion(1, 1), SchemaVersion(2, 0)),
            ("C", SchemaVersion(2, 0), SchemaVersion(3, 0)),
        ]
        assert execution_log == expected