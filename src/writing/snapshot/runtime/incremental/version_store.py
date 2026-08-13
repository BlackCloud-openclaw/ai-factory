# src/writing/snapshot/runtime/incremental/version_store.py
"""
B3.4: VersionStore — 版本元数据存储协议及内存实现
"""

from typing import Iterable, Protocol, runtime_checkable

from ..id import SnapshotId
from .version_manifest import VersionManifest
from .version_errors import VersionNotFoundError


@runtime_checkable
class VersionStore(Protocol):
    """版本元数据存储协议。"""

    def put(self, manifest: VersionManifest) -> None:
        """存储 VersionManifest（若已存在则覆盖）。"""
        ...

    def get(self, snapshot_id: SnapshotId) -> VersionManifest:
        """读取指定 snapshot_id 的 VersionManifest。"""
        ...

    def delete(self, snapshot_id: SnapshotId) -> None:
        """删除指定版本的 Manifest（不级联）。"""
        ...

    def list_ids(self) -> Iterable[SnapshotId]:
        """列出所有已知的 SnapshotId。"""
        ...


class MemoryVersionStore:
    """内存版本存储实现（最小实现，仅供测试）。"""

    def __init__(self):
        self._manifests: dict[SnapshotId, VersionManifest] = {}

    def put(self, manifest: VersionManifest) -> None:
        self._manifests[manifest.snapshot_id] = manifest

    def get(self, snapshot_id: SnapshotId) -> VersionManifest:
        if snapshot_id not in self._manifests:
            raise VersionNotFoundError(f"Version not found: {snapshot_id}")
        return self._manifests[snapshot_id]

    def delete(self, snapshot_id: SnapshotId) -> None:
        self._manifests.pop(snapshot_id, None)

    def list_ids(self) -> Iterable[SnapshotId]:
        return self._manifests.keys()