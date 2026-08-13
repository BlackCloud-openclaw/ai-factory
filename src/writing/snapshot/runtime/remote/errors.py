# src/writing/snapshot/runtime/remote/errors.py
"""
B4: Remote 存储层错误类型
"""

from ..exceptions import SnapshotStoreError


class RemoteStoreError(SnapshotStoreError):
    """远程存储层基类错误。"""
    pass


class ConcurrentModificationError(RemoteStoreError):
    """乐观锁冲突：版本已变更。"""
    pass


class RemoteConnectionError(RemoteStoreError):
    """远程服务连接失败。"""
    pass


class RemoteTimeoutError(RemoteStoreError):
    """远程请求超时。"""
    pass


class SnapshotHasChildrenError(RemoteStoreError):
    """Snapshot 仍有子版本，无法删除（force=False）。"""
    pass