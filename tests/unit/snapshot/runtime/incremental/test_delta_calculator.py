# tests/unit/snapshot/runtime/incremental/test_delta_calculator.py

import pytest

from src.writing.snapshot.runtime.chunking import Chunk
from src.writing.snapshot.runtime.incremental import ChunkSet, DeltaChunkSet, DeltaCalculator


class TestChunkSet:
    def test_creation_and_access(self):
        chunk1 = Chunk(chunk_id=1, payload=b"hello")
        chunk2 = Chunk(chunk_id=2, payload=b"world")
        cs = ChunkSet.from_mapping({1: chunk1, 2: chunk2})

        assert cs.get(1) == chunk1
        assert cs.get(2) == chunk2
        assert cs.get(3) is None
        assert len(cs) == 2
        assert set(cs.keys()) == {1, 2}
        assert set(cs.values()) == {chunk1, chunk2}
        assert dict(cs.items()) == {1: chunk1, 2: chunk2}

        # 测试 __contains__（统一使用 in）
        assert 1 in cs
        assert 3 not in cs

        # 测试 __iter__
        keys = [cid for cid in cs]
        assert set(keys) == {1, 2}

    def test_immutability_contract(self):
        cs = ChunkSet.from_mapping({1: Chunk(chunk_id=1, payload=b"test")})

        items = cs.items()
        # ItemsView 不可下标，应抛出 TypeError
        with pytest.raises(TypeError):
            items[0]  # type: ignore

        with pytest.raises(Exception):
            cs._chunks[1] = Chunk(chunk_id=1, payload=b"other")  # type: ignore

    def test_equality(self):
        cs1 = ChunkSet.from_mapping({1: Chunk(chunk_id=1, payload=b"a")})
        cs2 = ChunkSet.from_mapping({1: Chunk(chunk_id=1, payload=b"a")})
        cs3 = ChunkSet.from_mapping({1: Chunk(chunk_id=1, payload=b"b")})
        assert cs1 == cs2
        assert cs1 != cs3

    def test_empty_constant_value(self):
        from src.writing.snapshot.runtime.incremental.chunk_set import EMPTY_CHUNK_SET
        assert EMPTY_CHUNK_SET == ChunkSet.empty()
        assert len(EMPTY_CHUNK_SET) == 0


class TestDeltaChunkSet:
    def test_creation_and_access(self):
        delta = DeltaChunkSet(
            added_or_modified={1: Chunk(chunk_id=1, payload=b"new")},
            deleted=frozenset({2})
        )
        assert delta.is_empty() is False
        assert set(delta.keys()) == {1}
        assert set(delta.deleted) == {2}
        items = list(delta.items())
        assert items[0][1].payload == b"new"

    def test_invariant_disjoint(self):
        with pytest.raises(ValueError, match="appear in both"):
            DeltaChunkSet(
                added_or_modified={1: Chunk(chunk_id=1, payload=b"test")},
                deleted=frozenset({1})
            )

    def test_empty_delta(self):
        delta = DeltaChunkSet.empty()
        assert delta.is_empty() is True
        assert delta.added_or_modified == {}
        assert delta.deleted == frozenset()


class TestDeltaCalculator:
    def test_empty_delta(self):
        cs = ChunkSet.from_mapping({1: Chunk(chunk_id=1, payload=b"a")})
        delta = DeltaCalculator.compute_delta(cs, cs)
        assert delta.is_empty() is True

    def test_added_chunks(self):
        base = ChunkSet.empty()
        target = ChunkSet.from_mapping({1: Chunk(chunk_id=1, payload=b"new")})
        delta = DeltaCalculator.compute_delta(base, target)
        assert not delta.is_empty()
        assert set(delta.keys()) == {1}
        assert delta.deleted == frozenset()

    def test_deleted_chunks(self):
        base = ChunkSet.from_mapping({1: Chunk(chunk_id=1, payload=b"old")})
        target = ChunkSet.empty()
        delta = DeltaCalculator.compute_delta(base, target)
        assert set(delta.deleted) == {1}
        assert delta.added_or_modified == {}

    def test_modified_chunk(self):
        base = ChunkSet.from_mapping({1: Chunk(chunk_id=1, payload=b"old")})
        target = ChunkSet.from_mapping({1: Chunk(chunk_id=1, payload=b"new")})
        delta = DeltaCalculator.compute_delta(base, target)
        assert set(delta.keys()) == {1}
        assert delta.added_or_modified[1].payload == b"new"
        assert delta.deleted == frozenset()

    def test_unchanged_chunk(self):
        base = ChunkSet.from_mapping({1: Chunk(chunk_id=1, payload=b"same")})
        target = ChunkSet.from_mapping({1: Chunk(chunk_id=1, payload=b"same")})
        delta = DeltaCalculator.compute_delta(base, target)
        assert delta.is_empty() is True

    def test_mixed_changes(self):
        base = ChunkSet.from_mapping({
            1: Chunk(chunk_id=1, payload=b"a"),
            2: Chunk(chunk_id=2, payload=b"b"),
            3: Chunk(chunk_id=3, payload=b"c"),
        })
        target = ChunkSet.from_mapping({
            1: Chunk(chunk_id=1, payload=b"a"),          # unchanged
            2: Chunk(chunk_id=2, payload=b"b_modified"), # modified
            4: Chunk(chunk_id=4, payload=b"d"),          # added
            # 3 deleted
        })
        delta = DeltaCalculator.compute_delta(base, target)
        assert set(delta.keys()) == {2, 4}
        assert delta.added_or_modified[2].payload == b"b_modified"
        assert delta.added_or_modified[4].payload == b"d"
        assert set(delta.deleted) == {3}

    def test_order_independence(self):
        base = ChunkSet.from_mapping({
            5: Chunk(chunk_id=5, payload=b"a"),
            1: Chunk(chunk_id=1, payload=b"b"),
            8: Chunk(chunk_id=8, payload=b"c"),
        })
        target = ChunkSet.from_mapping({
            1: Chunk(chunk_id=1, payload=b"b"),
            5: Chunk(chunk_id=5, payload=b"a"),
            8: Chunk(chunk_id=8, payload=b"c"),
        })
        delta = DeltaCalculator.compute_delta(base, target)
        assert delta.is_empty() is True

    def test_round_trip_contract(self):
        base = ChunkSet.from_mapping({
            1: Chunk(chunk_id=1, payload=b"a"),
            2: Chunk(chunk_id=2, payload=b"b"),
        })
        target = ChunkSet.from_mapping({
            1: Chunk(chunk_id=1, payload=b"a_modified"),
            3: Chunk(chunk_id=3, payload=b"c"),
        })
        delta = DeltaCalculator.compute_delta(base, target)
        result = DeltaCalculator.apply_delta(base, delta)
        assert result == target

    def test_identity_contract(self):
        base = ChunkSet.from_mapping({1: Chunk(chunk_id=1, payload=b"a")})
        target = ChunkSet.from_mapping({1: Chunk(chunk_id=1, payload=b"b")})
        delta = DeltaCalculator.compute_delta(base, target)
        result = DeltaCalculator.apply_delta(base, delta)
        assert result == target

        # 对同一目标计算 delta 应为空
        delta_again = DeltaCalculator.compute_delta(target, target)
        assert delta_again.is_empty() is True

    def test_apply_delta_empty(self):
        base = ChunkSet.from_mapping({1: Chunk(chunk_id=1, payload=b"a")})
        delta = DeltaChunkSet.empty()
        result = DeltaCalculator.apply_delta(base, delta)
        assert result is base

    def test_apply_delta_fast_path(self):
        base = ChunkSet.empty()
        delta = DeltaChunkSet(added_or_modified={1: Chunk(chunk_id=1, payload=b"x")})
        result = DeltaCalculator.apply_delta(base, delta)
        expected = ChunkSet.from_mapping({1: Chunk(chunk_id=1, payload=b"x")})
        assert result == expected

    def test_apply_delta_deletion(self):
        base = ChunkSet.from_mapping({
            1: Chunk(chunk_id=1, payload=b"a"),
            2: Chunk(chunk_id=2, payload=b"b"),
        })
        delta = DeltaChunkSet(deleted=frozenset({1}))
        result = DeltaCalculator.apply_delta(base, delta)
        expected = ChunkSet.from_mapping({2: Chunk(chunk_id=2, payload=b"b")})
        assert result == expected

    def test_apply_delta_delete_nonexistent(self):
        """删除不存在的 chunk_id 应不产生异常（pop 安全）。"""
        base = ChunkSet.from_mapping({1: Chunk(chunk_id=1, payload=b"a")})
        delta = DeltaChunkSet(deleted=frozenset({2, 3}))
        result = DeltaCalculator.apply_delta(base, delta)
        expected = ChunkSet.from_mapping({1: Chunk(chunk_id=1, payload=b"a")})
        assert result == expected

    def test_apply_delta_modification(self):
        base = ChunkSet.from_mapping({1: Chunk(chunk_id=1, payload=b"a")})
        delta = DeltaChunkSet(added_or_modified={1: Chunk(chunk_id=1, payload=b"new")})
        result = DeltaCalculator.apply_delta(base, delta)
        expected = ChunkSet.from_mapping({1: Chunk(chunk_id=1, payload=b"new")})
        assert result == expected

    def test_immutability_of_inputs(self):
        base = ChunkSet.from_mapping({1: Chunk(chunk_id=1, payload=b"a")})
        target = ChunkSet.from_mapping({1: Chunk(chunk_id=1, payload=b"b")})
        DeltaCalculator.compute_delta(base, target)
        assert base.get(1).payload == b"a"
        assert target.get(1).payload == b"b"

    def test_apply_delta_does_not_mutate_inputs(self):
        base = ChunkSet.from_mapping({1: Chunk(chunk_id=1, payload=b"a")})
        delta = DeltaChunkSet(added_or_modified={1: Chunk(chunk_id=1, payload=b"new")})
        result = DeltaCalculator.apply_delta(base, delta)
        assert base.get(1).payload == b"a"
        assert delta.added_or_modified[1].payload == b"new"

    def test_delta_idempotence(self):
        base = ChunkSet.from_mapping({1: Chunk(chunk_id=1, payload=b"a")})
        target = ChunkSet.from_mapping({1: Chunk(chunk_id=1, payload=b"b")})
        delta = DeltaCalculator.compute_delta(base, target)
        result = DeltaCalculator.apply_delta(base, delta)
        assert result == target
        delta2 = DeltaCalculator.compute_delta(base, result)
        assert delta2 == delta