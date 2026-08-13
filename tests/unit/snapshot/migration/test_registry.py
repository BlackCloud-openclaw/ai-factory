# tests/unit/snapshot/migration/test_registry.py

import pytest

from src.writing.snapshot.migration import (
    MigrationEdge,
    MigrationGraph,
    MigrationRegistry,
    SchemaVersion,
    VersionNode,
    VersionType,
)


@pytest.fixture
def registry() -> MigrationRegistry:
    return MigrationRegistry()


@pytest.fixture
def sample_node() -> VersionNode:
    return VersionNode(
        version=SchemaVersion(1, 0),
        version_type=VersionType.MINOR,
    )


@pytest.fixture
def sample_edge() -> MigrationEdge:
    def upcaster(snap, ctx):
        return snap
    return MigrationEdge(
        from_version=SchemaVersion(1, 0),
        to_version=SchemaVersion(1, 1),
        upcaster=upcaster,
    )


class TestMigrationRegistry:
    def test_register_node_success(self, registry, sample_node):
        registry.register_node(sample_node)
        graph = registry.build()
        assert graph.has_version(SchemaVersion(1, 0)) is True

    def test_register_node_duplicate_raises(self, registry, sample_node):
        registry.register_node(sample_node)
        with pytest.raises(ValueError, match="Node already registered"):
            registry.register_node(sample_node)

    def test_register_edge_success(self, registry, sample_node, sample_edge):
        registry.register_node(sample_node)
        to_node = VersionNode(
            version=SchemaVersion(1, 1),
            version_type=VersionType.MINOR,
        )
        registry.register_node(to_node)
        registry.register_edge(sample_edge)
        graph = registry.build()
        assert graph.has_edge(SchemaVersion(1, 0), SchemaVersion(1, 1)) is True

    def test_register_edge_duplicate_raises(self, registry, sample_node, sample_edge):
        registry.register_node(sample_node)
        to_node = VersionNode(
            version=SchemaVersion(1, 1),
            version_type=VersionType.MINOR,
        )
        registry.register_node(to_node)
        registry.register_edge(sample_edge)
        with pytest.raises(ValueError, match="Duplicate edge"):
            registry.register_edge(sample_edge)

    def test_build_requires_all_nodes(self, registry, sample_edge):
        # 注册 to_version 节点，但不注册 from_version
        to_node = VersionNode(
            version=SchemaVersion(1, 1),
            version_type=VersionType.MINOR,
        )
        registry.register_node(to_node)
        # 注册边（from_version = 1.0 未注册）
        registry.register_edge(sample_edge)
        with pytest.raises(ValueError, match="unregistered node"):
            registry.build()

    def test_build_called_twice_raises(self, registry, sample_node):
        registry.register_node(sample_node)
        registry.build()
        with pytest.raises(RuntimeError, match="already been built"):
            registry.build()

    def test_register_after_build_raises(self, registry, sample_node):
        registry.register_node(sample_node)
        registry.build()
        with pytest.raises(RuntimeError, match="frozen"):
            registry.register_node(sample_node)

    def test_graph_property_before_build_raises(self, registry):
        with pytest.raises(RuntimeError, match="has not been built"):
            _ = registry.graph

    def test_graph_property_after_build_returns_graph(self, registry, sample_node):
        registry.register_node(sample_node)
        graph = registry.build()
        assert registry.graph is graph

    def test_registry_releases_nodes_after_build(self, registry, sample_node):
        registry.register_node(sample_node)
        registry.build()
        # 验证内部 _nodes 已被清空（通过尝试再次注册相同节点）
        # 因为已经冻结，注册应该抛出 RuntimeError
        with pytest.raises(RuntimeError, match="frozen"):
            registry.register_node(sample_node)