# src/writing/snapshot/runtime/remote/gc/retention.py
"""
B4.5: RetentionPolicy — 保留策略接口与实现
"""

from typing import Protocol, Iterable, List, Optional
from datetime import datetime

from ...id import SnapshotId
from ...incremental import VersionManifest


class RetentionPolicy(Protocol):
    """保留策略协议：从所有 Manifest 中选择保留锚点（retention anchors）。"""

    def select_roots(
        self,
        manifests: Iterable[VersionManifest],
    ) -> Iterable[SnapshotId]:
        """
        从所有 Manifest 中选择作为 GC 根的 Snapshot（保留锚点）。
        GC 将保留这些 Snapshot 及其所有祖先。
        """
        ...


class KeepAllPolicy:
    """保留所有 Snapshot（测试用）。"""

    def select_roots(
        self,
        manifests: Iterable[VersionManifest],
    ) -> Iterable[SnapshotId]:
        """
        返回保留锚点：所有没有父节点的 Snapshot（即原始 Base）。
        如果一个节点没有被任何其他节点引用，它就是树根。
        """
        manifest_list = list(manifests)
        if not manifest_list:
            return []

        parent_ids = {m.parent_id for m in manifest_list if m.parent_id is not None}
        roots = [m.snapshot_id for m in manifest_list if m.snapshot_id not in parent_ids]

        # 如果所有节点都有父节点（理论上是循环或单链），则选择 chain_length 最小的作为锚点
        if not roots and manifest_list:
            sorted_manifests = sorted(
                manifest_list,
                key=lambda m: m.metadata.get("chain_length", 0)
            )
            roots = [sorted_manifests[0].snapshot_id]

        return roots


class KeepLatestNPolicy:
    """
    保留最新的 N 个 Snapshot（及其祖先）。

    排序优先级：
        1. created_at（如果存在）
        2. sequence_number（如果存在）
        3. chain_length（fallback）
    """

    def __init__(self, n: int):
        if n <= 0:
            raise ValueError("n must be positive")
        self._n = n

    def _get_sort_key(self, manifest: VersionManifest):
        """生成排序键，确保最新优先。"""
        # 从 metadata 获取创建时间
        created_at = manifest.metadata.get("created_at")
        if created_at is not None:
            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at)
                except ValueError:
                    created_at = datetime.min
        else:
            created_at = datetime.min

        # 获取 sequence_number（如果存在）
        seq = manifest.metadata.get("sequence_number", 0)

        # 获取 chain_length
        chain_len = manifest.metadata.get("chain_length", 0)

        # 返回元组：先按 created_at 降序，再按 seq 降序，最后 chain_length 降序
        return (created_at, seq, chain_len)

    def select_roots(
        self,
        manifests: Iterable[VersionManifest],
    ) -> Iterable[SnapshotId]:
        manifest_list = list(manifests)
        if not manifest_list:
            return []

        # 按排序键降序排序（最新优先）
        sorted_manifests = sorted(
            manifest_list,
            key=self._get_sort_key,
            reverse=True,
        )

        # 取最新的 N 个作为锚点
        latest_n = sorted_manifests[:self._n]
        return [m.snapshot_id for m in latest_n]


class KeepSincePolicy:
    """保留指定时间之后创建的 Snapshot（及其祖先）。"""

    def __init__(self, since: datetime):
        self._since = since

    def select_roots(
        self,
        manifests: Iterable[VersionManifest],
    ) -> Iterable[SnapshotId]:
        result = []
        for m in manifests:
            created_at = m.metadata.get("created_at")
            if created_at is not None:
                if isinstance(created_at, str):
                    try:
                        created_at = datetime.fromisoformat(created_at)
                    except ValueError:
                        continue
                if isinstance(created_at, datetime) and created_at >= self._since:
                    result.append(m.snapshot_id)
        return result