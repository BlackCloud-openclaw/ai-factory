"""
Runtime 异常类型 - Phase 7A 收束
"""


class RuntimeError(Exception):
    """所有 Runtime 异常的基类"""
    pass


class UnknownSurfaceError(RuntimeError):
    """Builder 无法解析指定的 Surface ID"""
    pass


class DuplicateSurfaceError(RuntimeError):
    """Registry 检测到重复的 Surface ID"""
    pass


class SnapshotBuildError(RuntimeError):
    """Builder 构建 Snapshot 失败"""
    pass


class RegistryFrozenError(RuntimeError):
    """尝试修改已冻结的 Registry"""
    pass