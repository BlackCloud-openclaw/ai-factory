# src/writing/snapshot/runtime/exceptions.py
"""
B3: Runtime 自定义异常
"""


class SnapshotRuntimeError(Exception):
    """Runtime 层基类异常。"""
    pass


class SnapshotSerializationError(SnapshotRuntimeError):
    """序列化/反序列化失败。"""
    pass


class SnapshotStoreError(SnapshotRuntimeError):
    """存储层操作失败。"""
    pass


class SnapshotNotFoundError(SnapshotStoreError):
    """Snapshot 不存在。"""
    pass