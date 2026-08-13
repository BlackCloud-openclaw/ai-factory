# tests/unit/snapshot/migration/test_graph.py

import pytest

from src.writing.snapshot.migration import (
    MigrationEdge,
    MigrationGraph,
    PathStrategy,
    SchemaVersion,
    VersionNode,
    VersionType,
)


@pytest.fixture
def basic_graph() -> MigrationGraph:
    """基础图：1.0 -> 1.1 -> 2.0"""
    graph = MigrationGraph()

    v1_0 = VersionNode(
        version=SchemaVersion(1, 0),
        version_type=VersionType.MINOR,
    )
    v1_1 = VersionNode(
        version=SchemaVersion(1, 1),
        version_type=VersionType.MINOR,
    )
    v2_0 = VersionNode(
        version=SchemaVersion(2, 0),
        version_type=VersionType.MAJOR,
    )

    graph.add_node(v1_0)
    graph.add_node(v1_1)
    graph.add_node(v2_0)

    def upcaster(snap, ctx):
        return snap

    graph.add_edge(MigrationEdge(
        from_version=SchemaVersion(1, 0),
        to_version=SchemaVersion(1, 1),
        upcaster=upcaster,
    ))
    graph.add_edge(MigrationEdge(
        from_version=SchemaVersion(1, 1),
        to_version=SchemaVersion(2, 0),
        upcaster=upcaster,
    ))

    return graph


@pytest.fixture
def branch_graph() -> MigrationGraph:
    """
    分支图：1.0 -> 1.1 -> 2.0
              -> 1.5 -> 2.0
    """
    graph = MigrationGraph()

    for v in [
        SchemaVersion(1, 0),
        SchemaVersion(1, 1),
        SchemaVersion(1, 5),
        SchemaVersion(2, 0),
    ]:
        graph.add_node(VersionNode(
            version=v,
            version_type=VersionType.MINOR,
        ))

    def upcaster(snap, ctx):
        return snap

    graph.add_edge(MigrationEdge(
        from_version=SchemaVersion(1, 0),
        to_version=SchemaVersion(1, 1),
        upcaster=upcaster,
    ))
    graph.add_edge(MigrationEdge(
        from_version=SchemaVersion(1, 1),
        to_version=SchemaVersion(2, 0),
        upcaster=upcaster,
    ))
    graph.add_edge(MigrationEdge(
        from_version=SchemaVersion(1, 0),
        to_version=SchemaVersion(1, 5),
        upcaster=upcaster,
    ))
    graph.add_edge(MigrationEdge(
        from_version=SchemaVersion(1, 5),
        to_version=SchemaVersion(2, 0),
        upcaster=upcaster,
    ))

    return graph


# ================================================================
# 节点管理
# ================================================================

class TestAddNode:
    def test_add_node_success(self):
        graph = MigrationGraph()
        node = VersionNode(
            version=SchemaVersion(1, 0),
            version_type=VersionType.MINOR,
        )
        graph.add_node(node)
        assert graph.has_version(SchemaVersion(1, 0)) is True
        assert graph.get_node(SchemaVersion(1, 0)) is node

    def test_add_node_duplicate_raises(self):
        graph = MigrationGraph()
        node = VersionNode(
            version=SchemaVersion(1, 0),
            version_type=VersionType.MINOR,
        )
        graph.add_node(node)
        with pytest.raises(ValueError, match="Version already exists"):
            graph.add_node(node)

    def test_get_node_returns_none_for_missing(self):
        graph = MigrationGraph()
        assert graph.get_node(SchemaVersion(1, 0)) is None

    def test_get_all_versions(self):
        graph = MigrationGraph()
        graph.add_node(VersionNode(version=SchemaVersion(2, 0), version_type=VersionType.MAJOR))
        graph.add_node(VersionNode(version=SchemaVersion(1, 0), version_type=VersionType.MINOR))
        versions = graph.get_all_versions()
        assert versions == (SchemaVersion(1, 0), SchemaVersion(2, 0))


# ================================================================
# 边管理
# ================================================================

class TestAddEdge:
    def test_add_edge_success(self, basic_graph):
        assert basic_graph.has_edge(
            SchemaVersion(1, 0),
            SchemaVersion(1, 1),
        ) is True

    def test_add_edge_from_missing_node_raises(self):
        graph = MigrationGraph()
        graph.add_node(VersionNode(
            version=SchemaVersion(1, 1),
            version_type=VersionType.MINOR,
        ))

        def upcaster(snap, ctx):
            return snap

        with pytest.raises(ValueError, match="from_version node not found"):
            graph.add_edge(MigrationEdge(
                from_version=SchemaVersion(1, 0),
                to_version=SchemaVersion(1, 1),
                upcaster=upcaster,
            ))

    def test_add_edge_to_missing_node_raises(self):
        graph = MigrationGraph()
        graph.add_node(VersionNode(
            version=SchemaVersion(1, 0),
            version_type=VersionType.MINOR,
        ))

        def upcaster(snap, ctx):
            return snap

        with pytest.raises(ValueError, match="to_version node not found"):
            graph.add_edge(MigrationEdge(
                from_version=SchemaVersion(1, 0),
                to_version=SchemaVersion(1, 1),
                upcaster=upcaster,
            ))

    def test_add_edge_self_loop_raises(self):
        graph = MigrationGraph()
        graph.add_node(VersionNode(
            version=SchemaVersion(1, 0),
            version_type=VersionType.MINOR,
        ))

        def upcaster(snap, ctx):
            return snap

        with pytest.raises(ValueError, match="Self-loop"):
            graph.add_edge(MigrationEdge(
                from_version=SchemaVersion(1, 0),
                to_version=SchemaVersion(1, 0),
                upcaster=upcaster,
            ))


# ================================================================
# 路径查找
# ================================================================

class TestFindPath:
    def test_find_path_success(self, basic_graph):
        path = basic_graph.find_path(
            SchemaVersion(1, 0),
            SchemaVersion(2, 0),
        )
        assert path is not None
        assert len(path) == 2
        assert path[0].from_version == SchemaVersion(1, 0)
        assert path[0].to_version == SchemaVersion(1, 1)
        assert path[1].from_version == SchemaVersion(1, 1)
        assert path[1].to_version == SchemaVersion(2, 0)

    def test_find_path_source_equals_target(self, basic_graph):
        path = basic_graph.find_path(
            SchemaVersion(1, 0),
            SchemaVersion(1, 0),
        )
        assert path == []

    def test_find_path_no_path(self, basic_graph):
        path = basic_graph.find_path(
            SchemaVersion(1, 1),
            SchemaVersion(1, 0),  # 反向不存在
        )
        assert path is None

    def test_find_path_source_not_exists(self, basic_graph):
        path = basic_graph.find_path(
            SchemaVersion(3, 0),
            SchemaVersion(2, 0),
        )
        assert path is None

    def test_find_path_target_not_exists(self, basic_graph):
        path = basic_graph.find_path(
            SchemaVersion(1, 0),
            SchemaVersion(3, 0),
        )
        assert path is None

    def test_find_path_shortest_in_branch(self, branch_graph):
        # 分支图：1.0 -> 1.1 -> 2.0 (2跳) 或 1.0 -> 1.5 -> 2.0 (2跳)
        # 两条路径等长，SHORTEST 应该返回其中一条
        path = branch_graph.find_path(
            SchemaVersion(1, 0),
            SchemaVersion(2, 0),
            strategy=PathStrategy.SHORTEST,
        )
        assert path is not None
        assert len(path) == 2
        # 第一条边可能是 1.0->1.1 或 1.0->1.5，取决于 BFS 顺序
        assert path[0].from_version == SchemaVersion(1, 0)
        assert path[1].to_version == SchemaVersion(2, 0)

    def test_find_path_minor_first(self, branch_graph):
        # MINOR_FIRST 应优先选择 minor 升级
        path = branch_graph.find_path(
            SchemaVersion(1, 0),
            SchemaVersion(2, 0),
            strategy=PathStrategy.MINOR_FIRST,
        )
        assert path is not None
        assert len(path) == 2
        assert path[0].from_version == SchemaVersion(1, 0)
        assert path[0].to_version == SchemaVersion(1, 1)  # 优先 minor 而非 1.5

    def test_find_shortest_path_alias(self, basic_graph):
        path = basic_graph.find_shortest_path(
            SchemaVersion(1, 0),
            SchemaVersion(2, 0),
        )
        assert path is not None
        assert len(path) == 2


# ================================================================
# 循环检测
# ================================================================

class TestCycleDetection:
    def test_acyclic_graph_passes(self, basic_graph):
        basic_graph.validate_acyclic()  # 不应抛出异常
        assert basic_graph.has_cycle() is False


# ================================================================
# 拓扑排序
# ================================================================

class TestTopologicalOrder:
    def test_topological_order_acyclic(self, basic_graph):
        order = basic_graph.topological_order()
        # 1.0 -> 1.1 -> 2.0 或 1.0 -> 2.0? 取决于边
        assert len(order) == 3
        assert order[0] == SchemaVersion(1, 0)  # 最旧版本应在最前
        assert order[-1] == SchemaVersion(2, 0)  # 最新版本应在最后

# ================================================================
# 图信息
# ================================================================

class TestGraphInfo:
    def test_node_count(self, basic_graph):
        assert basic_graph.node_count() == 3

    def test_edge_count(self, basic_graph):
        assert basic_graph.edge_count() == 2

    def test_is_empty(self):
        graph = MigrationGraph()
        assert graph.is_empty() is True
        graph.add_node(VersionNode(
            version=SchemaVersion(1, 0),
            version_type=VersionType.MINOR,
        ))
        assert graph.is_empty() is False

    def test_get_edges_from(self, basic_graph):
        edges = basic_graph.get_edges_from(SchemaVersion(1, 0))
        assert len(edges) == 1
        assert edges[0].to_version == SchemaVersion(1, 1)

    def test_get_edges_from_empty(self, basic_graph):
        edges = basic_graph.get_edges_from(SchemaVersion(1, 1))
        assert len(edges) == 1
        assert edges[0].to_version == SchemaVersion(2, 0)

    def test_get_edges_to(self, basic_graph):
        edges = basic_graph.get_edges_to(SchemaVersion(1, 1))
        assert len(edges) == 1
        assert edges[0].from_version == SchemaVersion(1, 0)

        edges = basic_graph.get_edges_to(SchemaVersion(2, 0))
        assert len(edges) == 1
        assert edges[0].from_version == SchemaVersion(1, 1)
        
    # 新增/修改的测试方法

    def test_add_edge_duplicate_raises(self, basic_graph):
        def upcaster(snap, ctx):
            return snap
        edge = MigrationEdge(
            from_version=SchemaVersion(1, 0),
            to_version=SchemaVersion(1, 1),
            upcaster=upcaster,
        )
        with pytest.raises(ValueError, match="Migration edge already exists"):
            basic_graph.add_edge(edge)

    def test_get_edges_from_existing(self, basic_graph):
        edges = basic_graph.get_edges_from(SchemaVersion(1, 1))
        assert len(edges) == 1
        assert edges[0].to_version == SchemaVersion(2, 0)

    def test_get_edges_from_missing(self, basic_graph):
        edges = basic_graph.get_edges_from(SchemaVersion(9, 9))
        assert edges == ()