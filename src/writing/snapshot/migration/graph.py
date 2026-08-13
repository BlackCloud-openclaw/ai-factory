# src/writing/snapshot/migration/graph.py
"""
B1.3 MigrationGraph — 版本迁移拓扑图

职责：
- 管理版本节点（VersionNode）和迁移边（MigrationEdge）
- 提供路径查找（BFS + 策略）
- 检测循环并验证图合法性
- 提供拓扑排序

不依赖：
- 无 I/O、无网络、无数据库
- 无 CapabilityRegistry（Capability 在 Runtime 层组合）
"""

from __future__ import annotations

from collections import deque
from enum import Enum, auto
from typing import Optional

from .version import MigrationEdge, SchemaVersion, VersionNode


class PathStrategy(Enum):
    """路径查找策略。"""

    SHORTEST = auto()      # 最少跳数（BFS）
    MINOR_FIRST = auto()   # 同跳数时优先 minor 升级


class MigrationGraph:
    """版本迁移拓扑图（纯内存，不可变构建后只读）。"""

    def __init__(self) -> None:
        self._nodes: dict[SchemaVersion, VersionNode] = {}
        # 嵌套字典：from_version -> {to_version -> MigrationEdge}
        self._edges: dict[SchemaVersion, dict[SchemaVersion, MigrationEdge]] = {}

    # ================================================================
    # 节点管理
    # ================================================================

    def add_node(self, node: VersionNode) -> None:
        """
        添加版本节点。

        Raises:
            ValueError: 如果版本已存在
        """
        if node.version in self._nodes:
            raise ValueError(f"Version already exists: {node.version}")
        self._nodes[node.version] = node

    def has_version(self, version: SchemaVersion) -> bool:
        """检查版本是否已注册。"""
        return version in self._nodes

    def get_node(self, version: SchemaVersion) -> Optional[VersionNode]:
        """获取版本节点，不存在时返回 None。"""
        return self._nodes.get(version)

    def get_all_versions(self) -> tuple[SchemaVersion, ...]:
        """返回所有已注册版本的元组（按版本排序）。"""
        return tuple(sorted(self._nodes.keys()))

    # ================================================================
    # 边管理
    # ================================================================

    # 在 add_edge 中增加重复检测
    def add_edge(self, edge: MigrationEdge) -> None:
        if edge.from_version not in self._nodes:
            raise ValueError(f"from_version node not found: {edge.from_version}")
        if edge.to_version not in self._nodes:
            raise ValueError(f"to_version node not found: {edge.to_version}")
        if edge.from_version == edge.to_version:
            raise ValueError("Self-loop is not allowed")
        if not edge.from_version.is_upgrade_to(edge.to_version):
            raise ValueError(
                f"MigrationEdge requires strictly increasing versions: "
                f"{edge.from_version} → {edge.to_version}"
            )
        if (edge.from_version in self._edges
                and edge.to_version in self._edges[edge.from_version]):
            raise ValueError(
                f"Migration edge already exists: {edge.from_version} -> {edge.to_version}"
            )
        self._edges.setdefault(edge.from_version, {})[edge.to_version] = edge
    
    # topological_order 确定性
    def topological_order(self) -> list[SchemaVersion]:
        self.validate_acyclic()
        in_degree = {v: 0 for v in self._nodes}
        for edges in self._edges.values():
            for to_ver in edges:
                if to_ver in in_degree:
                    in_degree[to_ver] += 1

        queue = deque(sorted([v for v, deg in in_degree.items() if deg == 0]))
        result = []
        while queue:
            v = queue.popleft()
            result.append(v)
            for to_ver in sorted(self._edges.get(v, {}).keys()):
                in_degree[to_ver] -= 1
                if in_degree[to_ver] == 0:
                    queue.append(to_ver)

        if len(result) != len(self._nodes):
            raise RuntimeError("Graph has cycle, topological order not possible")
        return result

    def has_edge(self, from_version: SchemaVersion, to_version: SchemaVersion) -> bool:
        """检查从 from_version 到 to_version 的边是否存在。"""
        return (
            from_version in self._edges
            and to_version in self._edges[from_version]
        )

    def get_edge(
        self,
        from_version: SchemaVersion,
        to_version: SchemaVersion,
    ) -> Optional[MigrationEdge]:
        """获取边，不存在时返回 None。"""
        if from_version not in self._edges:
            return None
        return self._edges[from_version].get(to_version)

    def get_edges_from(
        self,
        version: SchemaVersion,
    ) -> tuple[MigrationEdge, ...]:
        """返回从该版本出发的所有边（按 to_version 排序）。"""
        if version not in self._edges:
            return ()
        edges = self._edges[version].values()
        return tuple(sorted(edges, key=lambda e: e.to_version))

    def get_edges_to(
        self,
        version: SchemaVersion,
    ) -> tuple[MigrationEdge, ...]:
        """返回指向该版本的所有边（按 from_version 排序）。"""
        result: list[MigrationEdge] = []
        for edges in self._edges.values():
            if version in edges:
                result.append(edges[version])
        return tuple(sorted(result, key=lambda e: e.from_version))

    # ================================================================
    # 路径查找
    # ================================================================

    def find_path(
        self,
        source: SchemaVersion,
        target: SchemaVersion,
        *,
        strategy: PathStrategy = PathStrategy.SHORTEST,
    ) -> Optional[list[MigrationEdge]]:
        """
        查找从 source 到 target 的迁移路径。

        Args:
            source: 起始版本
            target: 目标版本
            strategy: 路径策略
                - SHORTEST: 最少跳数（BFS）
                - MINOR_FIRST: 同跳数时优先 minor 升级

        Returns:
            迁移边列表（按执行顺序），若不存在路径则返回 None
        """
        if source == target:
            return []

        if source not in self._nodes or target not in self._nodes:
            return None

        # BFS 队列：(当前版本, 路径边列表)
        queue = deque([(source, [])])
        visited: set[SchemaVersion] = {source}

        while queue:
            current, path = queue.popleft()

            if current == target:
                return path

            # 获取当前版本的所有出边
            edges = list(self._edges.get(current, {}).values())

            # 根据策略排序
            if strategy == PathStrategy.MINOR_FIRST:
                edges.sort(
                    key=lambda e: 0 if e.from_version.is_minor_upgrade_to(e.to_version) else 1
                )
            # SHORTEST 不需要排序，BFS 天然保证最短

            for edge in edges:
                if edge.to_version not in visited:
                    visited.add(edge.to_version)
                    queue.append((edge.to_version, path + [edge]))

        return None

    def find_shortest_path(
        self,
        source: SchemaVersion,
        target: SchemaVersion,
    ) -> Optional[list[MigrationEdge]]:
        """同 find_path(..., strategy=PathStrategy.SHORTEST)。"""
        return self.find_path(source, target, strategy=PathStrategy.SHORTEST)

    # ================================================================
    # 图验证
    # ================================================================

    def validate_acyclic(self) -> None:
        """
        检查图中是否存在循环。

        Raises:
            RuntimeError: 如果检测到循环，错误信息包含循环路径
        """
        visited: set[SchemaVersion] = set()
        rec_stack: set[SchemaVersion] = set()

        def dfs(v: SchemaVersion, path: list[SchemaVersion]) -> None:
            visited.add(v)
            rec_stack.add(v)
            path.append(v)

            for to_ver in self._edges.get(v, {}).keys():
                if to_ver not in visited:
                    dfs(to_ver, path)
                elif to_ver in rec_stack:
                    # 找到循环
                    cycle_start = path.index(to_ver)
                    cycle = path[cycle_start:] + [to_ver]
                    cycle_str = " -> ".join(str(v) for v in cycle)
                    raise RuntimeError(f"Cycle detected: {cycle_str}")

            rec_stack.remove(v)
            path.pop()

        for node in self._nodes:
            if node not in visited:
                dfs(node, [])

    def has_cycle(self) -> bool:
        """检查图中是否存在循环（不抛出异常）。"""
        try:
            self.validate_acyclic()
            return False
        except RuntimeError:
            return True

    # ================================================================
    # 图信息
    # ================================================================

    def node_count(self) -> int:
        """返回节点数量。"""
        return len(self._nodes)

    def edge_count(self) -> int:
        """返回边数量。"""
        return sum(len(edges) for edges in self._edges.values())

    def is_empty(self) -> bool:
        """返回图是否为空。"""
        return len(self._nodes) == 0