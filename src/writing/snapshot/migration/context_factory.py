# src/writing/snapshot/migration/context_factory.py
"""
B2: MigrationContextFactory — 迁移上下文创建
"""

from typing import Protocol

from .version import MigrationContext


class MigrationContextFactory(Protocol):
    """创建 Migration 执行上下文。"""

    def create(self) -> MigrationContext:
        ...


class DefaultMigrationContextFactory:
    """默认实现（FixedClock + FixedRandom）。"""

    def create(self) -> MigrationContext:
        return MigrationContext()