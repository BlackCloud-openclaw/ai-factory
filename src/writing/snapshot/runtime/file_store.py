# src/writing/snapshot/runtime/file_store.py
"""
B3.1: FileSnapshotStore — 文件系统存储实现（生产默认）
"""

import os
from pathlib import Path
from typing import Iterable

from .exceptions import SnapshotNotFoundError, SnapshotStoreError
from .id import SnapshotId
from .path_strategy import DefaultSnapshotPathStrategy, SnapshotPathStrategy
from .protocols import SnapshotStore
from .record import SnapshotRecord
from .record_serializer import deserialize_record, serialize_record


class FileSnapshotStore:
    """
    文件系统存储实现。

    默认文件名: {snapshot_id.value}.snapshot
    可通过 strategy 自定义路径生成策略。
    """

    def __init__(
        self,
        base_dir: Path,
        strategy: SnapshotPathStrategy | None = None,
    ):
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._strategy = strategy or DefaultSnapshotPathStrategy(base_dir)

    def read(self, snapshot_id: SnapshotId) -> SnapshotRecord:
        path = self._strategy.path(snapshot_id)
        if not path.exists():
            raise SnapshotNotFoundError(f"Snapshot not found: {snapshot_id}")

        try:
            data = path.read_bytes()
            return deserialize_record(data)
        except OSError as e:
            raise SnapshotStoreError(f"Failed to read snapshot: {e}") from e

    def write(self, snapshot_id: SnapshotId, record: SnapshotRecord) -> None:
        path = self._strategy.path(snapshot_id)
        tmp_path = path.with_suffix(path.suffix + ".tmp")

        try:
            data = serialize_record(record)

            # 原子写入 + fsync（crash-safe）
            with open(tmp_path, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())

            tmp_path.replace(path)

        except OSError as e:
            raise SnapshotStoreError(f"Failed to write snapshot: {e}") from e
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def exists(self, snapshot_id: SnapshotId) -> bool:
        return self._strategy.path(snapshot_id).exists()

    def delete(self, snapshot_id: SnapshotId) -> None:
        path = self._strategy.path(snapshot_id)
        if path.exists():
            try:
                path.unlink()
            except OSError as e:
                raise SnapshotStoreError(f"Failed to delete snapshot: {e}") from e

    def list(self) -> Iterable[SnapshotId]:
        """列出所有 SnapshotId（通过扫描目录中的 .snapshot 文件）。"""
        extension = ".snapshot"
        for path in self._base_dir.glob(f"*{extension}"):
            try:
                # 文件名格式: {uuid}{extension}
                name = path.name[:-len(extension)] if path.name.endswith(extension) else path.name
                yield SnapshotId.from_string(name)
            except ValueError:
                # 忽略非 UUID 格式的文件
                continue