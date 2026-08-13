# src/writing/snapshot/migration/builtin/migrations/__init__.py
"""
B2: 内置迁移边注册
"""

from .v1_0_to_1_1 import register_v1_0_to_1_1
from .v1_1_to_2_0 import register_v1_1_to_2_0


def register_all_migrations(registry):
    """注册所有内置迁移边。"""
    register_v1_0_to_1_1(registry)
    register_v1_1_to_2_0(registry)