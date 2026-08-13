# src/writing/snapshot/migration/registry.py
"""
B2: MigrationRegistry — 迁移边注册与 Graph 构建（最终版）

生命周期：
1. register_node() / register_edge() 注册数据
2. build() 构建 Graph，校验合法性，释放注册数据
3. graph 属性获取已构建的 Graph（只读）
"""

from typing import Optional

from .graph import MigrationGraph
from .version import MigrationEdge, SchemaVersion, VersionNode


class MigrationRegistry:
    """
    迁移注册表。

    使用流程（建议）：
    1. 注册所有 VersionNode
    2. 注册所有 MigrationEdge
    3. build() 构建并冻结 MigrationGraph
    4. 通过 graph 属性访问 Graph

    注意：build() 后注册数据会被释放，Registry 仅保留 Graph。
    """

    def __init__(self):
        self._nodes: dict[SchemaVersion, VersionNode] = {}
        self._edges: dict[tuple[SchemaVersion, SchemaVersion], MigrationEdge] = {}
        self._frozen: bool = False
        self._graph: Optional[MigrationGraph] = None

    def register_node(self, node: VersionNode) -> None:
        """
        注册版本节点。

        Raises:
            RuntimeError: Registry 已冻结
            ValueError: 节点已存在
        """
        if self._frozen:
            raise RuntimeError("Registry is frozen after build()")
        if node.version in self._nodes:
            raise ValueError(f"Node already registered: {node.version}")
        self._nodes[node.version] = node

    def register_edge(self, edge: MigrationEdge) -> None:
        """
        注册迁移边。

        Raises:
            RuntimeError: Registry 已冻结
            ValueError: 边已存在
        """
        if self._frozen:
            raise RuntimeError("Registry is frozen after build()")
        key = (edge.from_version, edge.to_version)
        if key in self._edges:
            raise ValueError(
                f"Duplicate edge: {edge.from_version} -> {edge.to_version}"
            )
        self._edges[key] = edge

    def build(self) -> MigrationGraph:
        """
        构建并冻结 MigrationGraph。

        流程：
        1. 校验所有 Edge 引用的 Node 已注册
        2. 按版本排序 Node（确定性）
        3. 构建 Graph
        4. 释放 _nodes/_edges，仅保留 _graph

        Returns:
            构建好的 MigrationGraph

        Raises:
            RuntimeError: 已构建过
            ValueError: Edge 引用了未注册的节点
        """
        if self._frozen:
            raise RuntimeError("Registry has already been built")
        self._frozen = True

        # 校验所有 Edge 引用的 Node 已注册
        for (from_ver, to_ver) in self._edges:
            if from_ver not in self._nodes:
                raise ValueError(
                    f"Edge {from_ver} -> {to_ver} references "
                    f"unregistered node: {from_ver}"
                )
            if to_ver not in self._nodes:
                raise ValueError(
                    f"Edge {from_ver} -> {to_ver} references "
                    f"unregistered node: {to_ver}"
                )

        # 按版本排序（确定性）
        sorted_versions = sorted(self._nodes.keys())

        graph = MigrationGraph()
        for version in sorted_versions:
            graph.add_node(self._nodes[version])

        for edge in self._edges.values():
            graph.add_edge(edge)

        # 生命周期：释放注册数据，仅保留 Graph
        self._nodes.clear()
        self._edges.clear()
        self._graph = graph
        return graph

    @property
    def graph(self) -> MigrationGraph:
        """获取已构建的 MigrationGraph（只读）。"""
        if self._graph is None:
            raise RuntimeError("Registry has not been built yet")
        return self._graph