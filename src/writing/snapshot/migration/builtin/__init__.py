# src/writing/snapshot/migration/builtin/__init__.py
"""
B2: 内置迁移定义（Package API）

Bootstrap 通过 register_builtin_migrations() 统一注册所有内置迁移，
但不知道具体的 VersionNode、MigrationEdge 或 Upcaster 实现。
"""

from .register import register_builtin_migrations

__all__ = ["register_builtin_migrations"]