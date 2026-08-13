# src/writing/snapshot/runtime/remote/s3/errors.py
"""
B4.3: S3 错误类型映射
"""

from ..errors import RemoteStoreError, RemoteConnectionError, RemoteTimeoutError


class S3Error(RemoteStoreError):
    """S3 存储层基类错误。"""
    pass


class S3ConnectionError(RemoteConnectionError):
    """S3 连接失败。"""
    pass


class S3TimeoutError(RemoteTimeoutError):
    """S3 请求超时。"""
    pass


class S3NotFoundError(S3Error):
    """S3 对象不存在。"""
    pass


class S3ConflictError(S3Error):
    """S3 冲突（如条件写入失败）。"""
    pass