# src/writing/audit/trace_traversal.py
"""
Phase 10.2: TraceTraversal — 基于 DAG 的 Artifact 祖先闭包与 Lineage
"""

from typing import List, Set, Dict, Optional, Tuple
from uuid import UUID
from collections import deque

from .trace import ExecutionTrace, Artifact


class TraceTraversal:
    """
    基于 ExecutionTrace DAG 的遍历工具，支持祖先闭包和 Lineage。
    """

    def __init__(self, trace: ExecutionTrace):
        self._trace = trace
        self._adj = self._build_adjacency()
        self._sources = self._find_sources()
        self._sinks = self._find_sinks()

    def _build_adjacency(self) -> Dict[UUID, List[UUID]]:
        """构建 Artifact 邻接表（直接下游）。"""
        adj = {aid: [] for aid in self._trace.artifacts}
        for aid in self._trace.artifacts:
            for child in self._trace.children(aid):
                if child is not None:
                    adj[aid].append(child.artifact_id)
        return adj

    def _find_sources(self) -> List[UUID]:
        """找出所有源 Artifact（没有上游输入）。"""
        return [aid for aid in self._trace.artifacts if not self._trace.parents(aid)]

    def _find_sinks(self) -> List[UUID]:
        """找出所有终端 Artifact（没有下游输出）。"""
        return [aid for aid in self._trace.artifacts if not self._trace.children(aid)]

    def get_sources(self) -> List[UUID]:
        return self._sources

    def get_sinks(self) -> List[UUID]:
        return self._sinks

    def get_ancestor_closure(self, target: UUID) -> List[UUID]:
        """
        获取 target 的祖先闭包（包括 target 自身），
        返回拓扑顺序（从源到目标）。
        """
        ancestors = set()
        queue = deque([target])
        while queue:
            current = queue.popleft()
            if current in ancestors:
                continue
            ancestors.add(current)
            parents = self._trace.parents(current)
            for p in parents:
                if p is not None and p.artifact_id not in ancestors:
                    queue.append(p.artifact_id)
        # 拓扑排序：从源到目标
        nodes = list(ancestors)
        in_degree = {n: 0 for n in nodes}
        for u in nodes:
            for v in self._adj.get(u, []):
                if v in in_degree:
                    in_degree[v] += 1
        queue2 = deque(sorted([n for n, d in in_degree.items() if d == 0], key=lambda x: str(x)))
        ordered = []
        while queue2:
            u = queue2.popleft()
            ordered.append(u)
            for v in self._adj.get(u, []):
                if v in in_degree:
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        queue2.append(v)
        if len(ordered) < len(nodes):
            # 有环，回退到时间顺序
            time_map = {}
            for stage in self._trace.stages:
                for aid in stage.output_artifacts:
                    if aid in nodes:
                        time_map[aid] = stage.start_time
                for aid in stage.input_artifacts:
                    if aid in nodes:
                        time_map[aid] = stage.start_time
            ordered = sorted(nodes, key=lambda aid: time_map.get(aid, self._trace.start_time))
        return ordered

    def get_lineages_to_sink(self, source: UUID, sink: UUID) -> List[List[UUID]]:
        """
        获取从 source 到 sink 的所有路径（Lineages）。
        如果 DAG 较大，可能会较多，一般场景下数量可控。
        """
        if source == sink:
            return [[source]]
        paths = []
        self._dfs_paths(source, sink, [source], set(), paths)
        return paths

    def _dfs_paths(self, current: UUID, target: UUID, path: List[UUID], visited: Set[UUID], result: List[List[UUID]]):
        if current == target:
            result.append(path.copy())
            return
        for neighbor in self._adj.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                self._dfs_paths(neighbor, target, path, visited, result)
                path.pop()
                visited.remove(neighbor)

    def get_artifact(self, aid: UUID) -> Optional[Artifact]:
        return self._trace.get_artifact(aid)