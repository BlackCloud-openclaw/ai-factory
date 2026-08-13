# src/bootstrap/services.py
"""
B2: Snapshot Service Container

Bootstrap 返回此容器，而非单个 Loader。
未来扩展（Writer、Storage、Repository）只需在此添加字段。
"""

from dataclasses import dataclass

from src.writing.snapshot import SnapshotLoader
from src.writing.snapshot.migration import MigrationRegistry, MigrationRuntime


@dataclass(frozen=True)
class SnapshotServices:
    """Snapshot 子系统的所有服务（由 Composition Root 组装）。"""

    loader: SnapshotLoader
    runtime: MigrationRuntime
    registry: MigrationRegistry

    # 未来扩展预留：
    # writer: SnapshotWriter
    # storage: SnapshotStorage
    # repository: SnapshotRepository