# src/writing/snapshot/runtime/remote/gc/reachability.py
"""
B4.5: ReachabilityAnalyzer — 构建可达性图
"""

from typing import Optional, Set, Dict

from ...id import SnapshotId
from ...incremental import VersionManifest, VersionStore, VersionNotFoundError
from .capability import ChunkEnumerator
from .models import ChunkRef, ReachabilityGraph
from .retention import RetentionPolicy, KeepAllPolicy
from .errors import GCInconsistentError


class ReachabilityAnalyzer:
    """
    可达性分析器。

    从 VersionStore 加载所有 Manifest，根据 RetentionPolicy 选择根节点，
    然后沿着 parent_id 链遍历，收集所有可达的 Snapshot 和 Chunk。

    Chunk 引用从 ChunkEnumerator 按需获取（O(1) 查询）。
    """

    def __init__(
        self,
        version_store: VersionStore,
        chunk_enumerator: ChunkEnumerator,
        retention_policy: Optional[RetentionPolicy] = None,
    ):
        self._version_store = version_store
        self._chunk_enumerator = chunk_enumerator
        self._retention_policy = retention_policy or KeepAllPolicy()
        self._visited_global: Set[SnapshotId] = set()

    def analyze(self) -> ReachabilityGraph:
        """构建可达性图。"""
        all_ids = list(self._version_store.list_ids())
        manifests: Dict[SnapshotId, VersionManifest] = {}
        for sid in all_ids:
            try:
                manifests[sid] = self._version_store.get(sid)
            except VersionNotFoundError:
                continue

        if not manifests:
            return ReachabilityGraph()

        roots = set(self._retention_policy.select_roots(manifests.values()))

        reachable_snapshots: Set[SnapshotId] = set()
        reachable_chunks: Set[ChunkRef] = set()
        self._visited_global = set()

        for root_id in roots:
            self._walk_chain(
                root_id,
                manifests,
                reachable_snapshots,
                reachable_chunks,
            )

        return ReachabilityGraph(
            reachable_snapshots=frozenset(reachable_snapshots),
            reachable_chunks=frozenset(reachable_chunks),
        )

    def _walk_chain(
        self,
        snapshot_id: SnapshotId,
        manifests: Dict[SnapshotId, VersionManifest],
        reachable_snapshots: Set[SnapshotId],
        reachable_chunks: Set[ChunkRef],
    ) -> None:
        """沿 parent 链向上遍历。"""
        current = snapshot_id
        visited_local: Set[SnapshotId] = set()

        while current is not None:
            if current in self._visited_global:
                reachable_snapshots.add(current)
                current = None
                break

            if current in visited_local:
                raise GCInconsistentError(
                    f"Cycle detected in version chain: {current}"
                )
            visited_local.add(current)

            reachable_snapshots.add(current)

            for chunk_ref in self._chunk_enumerator.list_chunks(current):
                reachable_chunks.add(chunk_ref)

            manifest = manifests.get(current)
            if manifest is None:
                raise GCInconsistentError(
                    f"Manifest for {current} not found (chain broken)"
                )

            current = manifest.parent_id

        self._visited_global.update(visited_local)