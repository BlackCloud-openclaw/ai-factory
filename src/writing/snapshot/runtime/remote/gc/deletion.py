# src/writing/snapshot/runtime/remote/gc/deletion.py
"""
B4.5/B4.6: DeletionPlanner — 计算待删除的 Chunk（纯函数）
"""

from typing import Iterable

from .models import ChunkRef, ChunkMetadata, DeletionCandidate, DeletionPlan


class DeletionPlanner:
    """
    删除规划器（纯函数）。

    计算 all_physical_chunks - reachable_chunks 的差集，
    生成待删除的 Chunk 列表。
    输入为 ChunkMetadata 列表，输出携带完整元数据。
    """

    def plan(
        self,
        all_chunks: Iterable[ChunkMetadata],
        reachable_chunks: Iterable[ChunkRef],
    ) -> DeletionPlan:
        """
        计算待删除的 Chunk。

        Args:
            all_chunks: 所有物理存在的 Chunk（含元数据）
            reachable_chunks: 所有可达的 Chunk 引用

        Returns:
            DeletionPlan: 待删除的 Chunk 列表（含完整元数据）
        """
        reachable_set = set(reachable_chunks)

        candidates = []
        for metadata in all_chunks:
            if metadata.chunk_ref not in reachable_set:
                candidates.append(DeletionCandidate(metadata=metadata))

        return DeletionPlan(candidates=tuple(candidates))