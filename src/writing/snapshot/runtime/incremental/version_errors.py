# src/writing/snapshot/runtime/incremental/version_errors.py
"""
B3.4: 版本链相关异常
"""

from ..exceptions import SnapshotRuntimeError


class VersionError(SnapshotRuntimeError):
    """版本系统基类异常。"""
    pass


class VersionNotFoundError(VersionError):
    """指定的 SnapshotId 不存在。"""
    pass


class VersionCycleError(VersionError):
    """检测到版本循环依赖。"""
    pass


class VersionChainTooDeepError(VersionError):
    """版本链超出最大允许深度。"""
    pass