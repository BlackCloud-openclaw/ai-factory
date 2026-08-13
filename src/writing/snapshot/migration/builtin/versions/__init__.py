# src/writing/snapshot/migration/builtin/versions/__init__.py
"""
B2: 内置版本节点注册
"""

from .v1_0 import register_v1_0
from .v1_1 import register_v1_1
from .v2_0 import register_v2_0


def register_all_versions(registry):
    """注册所有内置版本节点。"""
    register_v1_0(registry)
    register_v1_1(registry)
    register_v2_0(registry)