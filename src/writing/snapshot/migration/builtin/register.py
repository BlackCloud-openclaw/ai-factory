# src/writing/snapshot/migration/builtin/register.py
"""
B2: 统一注册入口

Bootstrap 调用此函数注册所有内置迁移。
"""

from ..registry import MigrationRegistry
from .versions import register_all_versions
from .migrations import register_all_migrations


def register_builtin_migrations(registry: MigrationRegistry) -> None:
    """注册所有内置的版本节点和迁移边。"""
    register_all_versions(registry)
    register_all_migrations(registry)