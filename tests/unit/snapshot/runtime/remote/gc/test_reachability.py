# tests/unit/snapshot/runtime/remote/gc/test_reachability.py

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

from src.writing.snapshot.runtime import SnapshotId
from src.writing.snapshot.runtime.incremental import (
    VersionManifest,
    MemoryVersionStore,
    VersionNotFoundError,
)
from src.writing.snapshot.runtime.remote.gc import (
    ReachabilityAnalyzer,
    KeepAllPolicy,
    KeepLatestNPolicy,
    KeepSincePolicy,
    GCInconsistentError,
    ChunkRef,  # 添加导入
)
from src.writing.snapshot.runtime.remote.gc.capability import ChunkEnumerator


class TestReachabilityAnalyzer:
    def setup_method(self):
        self.version_store = MemoryVersionStore()
        self.chunk_enumerator = Mock(spec=ChunkEnumerator)
        self.chunk_enumerator.list_chunks.return_value = []
        self.chunk_enumerator.list_all_chunks.return_value = []

    def _create_manifest(self, sid: SnapshotId, parent_id: SnapshotId | None = None, metadata=None):
        meta = metadata or {}
        manifest = VersionManifest(
            snapshot_id=sid,
            parent_id=parent_id,
            metadata=meta,
        )
        self.version_store.put(manifest)
        return manifest

    def test_single_snapshot(self):
        sid = SnapshotId.new()
        self._create_manifest(sid)
        self.chunk_enumerator.list_chunks.return_value = [
            ChunkRef(sid, 1),
            ChunkRef(sid, 2),
            ChunkRef(sid, 3)
        ]

        analyzer = ReachabilityAnalyzer(self.version_store, self.chunk_enumerator)
        graph = analyzer.analyze()

        assert sid in graph.reachable_snapshots
        assert len(graph.reachable_snapshots) == 1
        assert len(graph.reachable_chunks) == 3
        assert ChunkRef(sid, 1) in graph.reachable_chunks
        assert ChunkRef(sid, 2) in graph.reachable_chunks
        assert ChunkRef(sid, 3) in graph.reachable_chunks

    def test_chain_of_snapshots(self):
        ids = [SnapshotId.new() for _ in range(3)]
        self._create_manifest(ids[0])
        self._create_manifest(ids[1], parent_id=ids[0])
        self._create_manifest(ids[2], parent_id=ids[1])

        def list_chunks_side_effect(sid):
            if sid == ids[0]:
                return [ChunkRef(ids[0], 1)]
            elif sid == ids[1]:
                return [ChunkRef(ids[1], 2)]
            elif sid == ids[2]:
                return [ChunkRef(ids[2], 3)]
            return []

        self.chunk_enumerator.list_chunks.side_effect = list_chunks_side_effect

        analyzer = ReachabilityAnalyzer(self.version_store, self.chunk_enumerator)
        graph = analyzer.analyze()

        assert ids[0] in graph.reachable_snapshots
        assert ids[1] in graph.reachable_snapshots
        assert ids[2] in graph.reachable_snapshots
        assert len(graph.reachable_snapshots) == 3
        assert len(graph.reachable_chunks) == 3

    def test_keep_latest_n_policy(self):
        ids = [SnapshotId.new() for _ in range(5)]
        now = datetime.now()
        for i in range(5):
            self._create_manifest(
                ids[i],
                parent_id=ids[i-1] if i > 0 else None,
                metadata={"chain_length": i+1, "created_at": now - timedelta(days=5-i)},
            )
        self.chunk_enumerator.list_chunks.return_value = []

        policy = KeepLatestNPolicy(n=2)
        analyzer = ReachabilityAnalyzer(self.version_store, self.chunk_enumerator, retention_policy=policy)
        graph = analyzer.analyze()

        assert ids[4] in graph.reachable_snapshots
        assert ids[3] in graph.reachable_snapshots
        assert ids[0] in graph.reachable_snapshots
        assert ids[1] in graph.reachable_snapshots
        assert ids[2] in graph.reachable_snapshots

    def test_keep_since_policy(self):
        now = datetime.now()
        ids = [SnapshotId.new() for _ in range(3)]
        self._create_manifest(
            ids[0],
            metadata={"created_at": now - timedelta(days=10)}
        )
        self._create_manifest(
            ids[1],
            metadata={"created_at": now - timedelta(days=5)}
        )
        self._create_manifest(
            ids[2],
            metadata={"created_at": now - timedelta(days=1)}
        )
        self.chunk_enumerator.list_chunks.return_value = []

        policy = KeepSincePolicy(since=now - timedelta(days=2))
        analyzer = ReachabilityAnalyzer(self.version_store, self.chunk_enumerator, retention_policy=policy)
        graph = analyzer.analyze()

        assert ids[2] in graph.reachable_snapshots
        assert ids[1] not in graph.reachable_snapshots
        assert ids[0] not in graph.reachable_snapshots

    def test_detect_cycle(self):
        a = SnapshotId.new()
        b = SnapshotId.new()
        self._create_manifest(a, parent_id=b)
        self._create_manifest(b, parent_id=a)
        self.chunk_enumerator.list_chunks.return_value = []

        analyzer = ReachabilityAnalyzer(self.version_store, self.chunk_enumerator)
        with pytest.raises(GCInconsistentError, match="Cycle detected"):
            analyzer.analyze()

    def test_broken_chain(self):
        a = SnapshotId.new()
        b = SnapshotId.new()
        c = SnapshotId.new()
        self._create_manifest(a, parent_id=None)
        self._create_manifest(c, parent_id=b)

        self.chunk_enumerator.list_chunks.return_value = []

        analyzer = ReachabilityAnalyzer(self.version_store, self.chunk_enumerator)
        with pytest.raises(GCInconsistentError, match="Manifest for .* not found"):
            analyzer.analyze()

    def test_global_visited_cache(self):
        ids = [SnapshotId.new() for _ in range(3)]
        self._create_manifest(ids[0])
        self._create_manifest(ids[1], parent_id=ids[0])
        self._create_manifest(ids[2], parent_id=ids[0])

        def list_chunks_side_effect(sid):
            if sid == ids[0]:
                return [ChunkRef(ids[0], 1), ChunkRef(ids[0], 2)]
            elif sid == ids[1]:
                return [ChunkRef(ids[1], 3)]
            elif sid == ids[2]:
                return [ChunkRef(ids[2], 4)]
            return []

        self.chunk_enumerator.list_chunks.side_effect = list_chunks_side_effect

        analyzer = ReachabilityAnalyzer(self.version_store, self.chunk_enumerator)
        graph = analyzer.analyze()

        assert ids[0] in graph.reachable_snapshots
        assert ids[1] in graph.reachable_snapshots
        assert ids[2] in graph.reachable_snapshots
        assert ChunkRef(ids[0], 1) in graph.reachable_chunks
        assert ChunkRef(ids[0], 2) in graph.reachable_chunks
        assert ChunkRef(ids[1], 3) in graph.reachable_chunks
        assert ChunkRef(ids[2], 4) in graph.reachable_chunks