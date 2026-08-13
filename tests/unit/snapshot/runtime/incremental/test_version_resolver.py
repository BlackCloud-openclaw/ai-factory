# tests/unit/snapshot/runtime/incremental/test_version_resolver.py

import pytest

from src.writing.snapshot.runtime import SnapshotId
from src.writing.snapshot.runtime.incremental import (
    VersionManifest,
    VersionChain,
    MemoryChunkRepository,
    VersionResolver,
    VersionNotFoundError,
    VersionCycleError,
    VersionChainTooDeepError,
    ChunkSet,
)


class TestVersionChain:
    def test_chain_properties(self):
        ids = [SnapshotId.new() for _ in range(3)]
        chain = VersionChain(tuple(ids))
        assert chain.base == ids[0]
        assert chain.latest == ids[-1]
        assert chain.depth == 3
        assert len(chain) == 3
        assert list(chain) == ids
        assert chain[0] == ids[0]
        assert chain[-1] == ids[-1]
        assert ids[0] in chain

    def test_empty_chain_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            VersionChain(())


class TestVersionResolver:
    def setup_method(self):
        self.repo = MemoryChunkRepository()

    def test_resolve_single_version(self):
        sid = SnapshotId.new()
        self.repo.save_version(sid, ChunkSet.empty(), parent_id=None)
        resolver = VersionResolver(self.repo)
        chain = resolver.resolve_chain(sid)
        assert chain.base == sid
        assert chain.latest == sid
        assert chain.depth == 1
        assert list(chain) == [sid]

    def test_resolve_two_version_chain(self):
        base = SnapshotId.new()
        child = SnapshotId.new()
        self.repo.save_version(base, ChunkSet.empty(), parent_id=None)
        self.repo.save_version(child, ChunkSet.empty(), parent_id=base)

        resolver = VersionResolver(self.repo)
        chain = resolver.resolve_chain(child)
        assert list(chain) == [base, child]
        assert chain.base == base
        assert chain.latest == child

    def test_resolve_three_version_chain(self):
        ids = [SnapshotId.new() for _ in range(3)]
        self.repo.save_version(ids[0], ChunkSet.empty(), parent_id=None)
        self.repo.save_version(ids[1], ChunkSet.empty(), parent_id=ids[0])
        self.repo.save_version(ids[2], ChunkSet.empty(), parent_id=ids[1])

        resolver = VersionResolver(self.repo)
        chain = resolver.resolve_chain(ids[2])
        assert list(chain) == ids

    def test_resolve_not_found(self):
        sid = SnapshotId.new()
        resolver = VersionResolver(self.repo)
        with pytest.raises(VersionNotFoundError):
            resolver.resolve_chain(sid)

    def test_resolve_cycle_detection(self):
        a = SnapshotId.new()
        b = SnapshotId.new()
        c = SnapshotId.new()

        self.repo.save_version(a, ChunkSet.empty(), parent_id=c)
        self.repo.save_version(b, ChunkSet.empty(), parent_id=a)
        self.repo.save_version(c, ChunkSet.empty(), parent_id=b)

        resolver = VersionResolver(self.repo)
        with pytest.raises(VersionCycleError, match="Cycle detected"):
            resolver.resolve_chain(a)

    def test_max_depth_exceeded(self):
        ids = [SnapshotId.new() for _ in range(33)]
        self.repo.save_version(ids[0], ChunkSet.empty(), parent_id=None)
        for i in range(1, 33):
            self.repo.save_version(ids[i], ChunkSet.empty(), parent_id=ids[i - 1])

        resolver = VersionResolver(self.repo, max_depth=32)
        with pytest.raises(VersionChainTooDeepError, match="exceeds max_depth"):
            resolver.resolve_chain(ids[-1])

    def test_missing_mid_chain_version(self):
        a = SnapshotId.new()
        b = SnapshotId.new()
        c = SnapshotId.new()

        self.repo.save_version(a, ChunkSet.empty(), parent_id=None)
        # b 缺失
        self.repo.save_version(c, ChunkSet.empty(), parent_id=b)

        resolver = VersionResolver(self.repo)
        with pytest.raises(VersionNotFoundError):
            resolver.resolve_chain(c)

    def test_chain_depth_fast_path(self):
        ids = [SnapshotId.new() for _ in range(5)]
        self.repo.save_version(ids[0], ChunkSet.empty(), parent_id=None)
        for i in range(1, 5):
            self.repo.save_version(ids[i], ChunkSet.empty(), parent_id=ids[i - 1])

        resolver = VersionResolver(self.repo)
        chain = resolver.resolve_chain(ids[4])
        assert chain.depth == 5
        assert chain.base == ids[0]
        assert chain.latest == ids[4]