# src/writing/snapshot/migration/runtime.py
"""
B1.4 MigrationRuntime — 迁移执行引擎

职责：
- 使用 MigrationGraph 查找路径
- 按顺序执行 Upcaster 链
- 验证输入输出完整性
- 保持不可变边界
"""

from __future__ import annotations

from typing import Optional, Protocol

from .graph import MigrationGraph, PathStrategy
from .raw_snapshot import RawSnapshot
from .version import MigrationContext, SchemaVersion
from .errors import MigrationPathNotFoundError, MigrationExecutionError


class MigrationObserver(Protocol):
    """
    预留的观测接口（Release C 实现）。

    B1.4 不触发任何回调，仅作为扩展点预留。
    未来 Release C 可定义：
    - on_migration_started(source, target)
    - on_edge_completed(edge, duration_ms)
    - on_migration_finished(result)
    - on_migration_failed(error)
    """
    pass


class MigrationRuntime:
    """
    迁移执行引擎。

    Runtime 是纯执行器，不持有迁移状态。
    - 每次 migrate() 调用独立执行
    - 输入 RawSnapshot 不变
    - 输出新的 RawSnapshot
    - observer 为 Release C 预留（日志/指标）
    """

    def __init__(
        self,
        graph: MigrationGraph,
        *,
        strategy: PathStrategy = PathStrategy.MINOR_FIRST,
        observer: Optional[MigrationObserver] = None,
    ):
        """
        Args:
            graph: 已构建的 MigrationGraph
            strategy: 路径查找策略，默认 MINOR_FIRST
            observer: 预留扩展点（Release C 实现），B1.4 不触发
        """
        self._graph = graph
        self._strategy = strategy
        self._observer = observer

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
            TypeError: snapshot 不是 RawSnapshot 或 context 不是 MigrationContext
            MigrationPathNotFoundError: 无迁移路径
            MigrationExecutionError: Upcaster 执行失败
        """
        # 1. 输入类型校验
        if not isinstance(snapshot, RawSnapshot):
            raise TypeError(
                f"snapshot must be RawSnapshot, got {type(snapshot).__name__}"
            )
        if not isinstance(context, MigrationContext):
            raise TypeError(
                f"context must be MigrationContext, got {type(context).__name__}"
            )

        source = snapshot.schema_version

        # 2. 快速路径：已为目标版本
        if source == target_version:
            return snapshot

        # 3. 查找路径
        path = self._graph.find_path(
            source,
            target_version,
            strategy=self._strategy,
        )
        if path is None:
            raise MigrationPathNotFoundError(
                f"No migration path from {source} to {target_version}"
            )

        # 4. 顺序执行 Upcaster 链
        current = snapshot
        for edge in path:
            try:
                result = edge.upcaster(current, context)
            except Exception as exc:
                raise MigrationExecutionError(
                    f"Migration failed at {edge.from_version} → {edge.to_version}"
                ) from exc

            # 返回值类型校验（强制 RawSnapshot）
            if not isinstance(result, RawSnapshot):
                raise MigrationExecutionError(
                    f"Upcaster must return RawSnapshot, got {type(result).__name__}"
                )

            current = result

        # 5. 最终版本一致性校验
        if current.schema_version != target_version:
            raise MigrationExecutionError(
                f"Migration finished with incorrect version: "
                f"expected {target_version}, got {current.schema_version}"
            )

        return current