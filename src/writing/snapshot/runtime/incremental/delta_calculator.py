# src/writing/snapshot/runtime/incremental/delta_calculator.py
"""
B3.4: DeltaCalculator — 计算两个 ChunkSet 之间的差异，并提供应用接口

ADR-B3-63: ChunkSet 的演化只能通过 DeltaCalculator.apply_delta() 完成。
"""

from .chunk_set import ChunkSet
from .delta_chunk_set import DeltaChunkSet
from ..chunking import Chunk


class DeltaCalculator:
    """
    纯函数计算两个 ChunkSet 之间的差异。

    实现约束：
        - O(n) 时间
        - O(k) 空间
        - 幂等性：compute_delta(base, base) -> 空 DeltaChunkSet
        - 纯函数：不修改输入对象
    """

    @classmethod
    def compute_delta(cls, base: ChunkSet, target: ChunkSet) -> DeltaChunkSet:
        """
        计算从 base 到 target 的差异。
        """
        added_or_modified: dict[int, Chunk] = {}
        deleted: set[int] = set()

        # 快速路径
        if len(target) == 0:
            deleted = set(base.keys())
            return DeltaChunkSet(added_or_modified={}, deleted=frozenset(deleted))

        if len(base) == 0:
            for cid, chunk in target.items():
                added_or_modified[cid] = chunk
            return DeltaChunkSet(added_or_modified=added_or_modified, deleted=frozenset())

        # 直接使用 base.get()，无需额外复制 base_chunks
        target_keys = set(target.keys())

        # 遍历 target，识别新增/修改
        for cid, target_chunk in target.items():
            base_chunk = base.get(cid)
            if base_chunk is None:
                added_or_modified[cid] = target_chunk
            elif base_chunk != target_chunk:
                added_or_modified[cid] = target_chunk

        # 识别删除（base 中存在但 target 中缺失）
        for cid in base.keys():
            if cid not in target_keys:
                deleted.add(cid)

        return DeltaChunkSet(
            added_or_modified=added_or_modified,
            deleted=frozenset(deleted),
        )

    @classmethod
    def apply_delta(cls, base: ChunkSet, delta: DeltaChunkSet) -> ChunkSet:
        """
        将 Delta 应用到 Base 上，生成新的 ChunkSet。

        唯一允许修改 ChunkSet 状态的接口。
        """
        if delta.is_empty():
            return base

        # 快速路径：base 为空且无删除，直接构造
        if len(base) == 0 and not delta.deleted:
            return ChunkSet.from_mapping(delta.added_or_modified)

        result = dict(base.items())
        for cid, chunk in delta.items():
            result[cid] = chunk
        for cid in delta.deleted:
            result.pop(cid, None)

        return ChunkSet.from_mapping(result)