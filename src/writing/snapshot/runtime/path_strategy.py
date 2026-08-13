# src/writing/snapshot/runtime/path_strategy.py
"""
B3.1: SnapshotPathStrategy — 路径生成策略
"""

from pathlib import Path
from typing import Protocol

from .id import SnapshotId


class SnapshotPathStrategy(Protocol):
    """路径生成策略协议。"""

    def path(self, snapshot_id: SnapshotId) -> Path:
        """根据 SnapshotId 生成存储路径。"""
        ...


class DefaultSnapshotPathStrategy:
    """默认路径策略：{base_dir}/{uuid}.snapshot"""

    def __init__(self, base_dir: Path, extension: str = ".snapshot"):
        self._base_dir = Path(base_dir)
        self._extension = extension

    def path(self, snapshot_id: SnapshotId) -> Path:
        return self._base_dir / f"{snapshot_id.value}{self._extension}"