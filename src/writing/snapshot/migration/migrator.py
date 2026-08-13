# src/writing/snapshot/migration/migrator.py
"""
B2: SnapshotMigrator Protocol — Loader 的唯一迁移依赖
"""

from typing import Protocol

from .raw_snapshot import RawSnapshot
from .version import MigrationContext, SchemaVersion


class SnapshotMigrator(Protocol):
    """迁移执行器协议。Loader 只依赖此接口，不依赖具体实现。"""

    def migrate(
        self,
        snapshot: RawSnapshot,
        target_version: SchemaVersion,
        context: MigrationContext,
    ) -> RawSnapshot:
        """
        执行版本迁移。

        Args:
            snapshot: 源快照（不可变）
            target_version: 目标版本
            context: 确定性执行上下文

        Returns:
            迁移后的新 RawSnapshot

        Raises:
            MigrationPathNotFoundError: 无迁移路径
            MigrationExecutionError: Upcaster 执行失败
        """
        ...