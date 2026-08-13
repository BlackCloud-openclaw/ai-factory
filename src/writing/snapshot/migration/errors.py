# src/writing/snapshot/migration/errors.py
"""
B1.4 Migration Runtime 自定义异常类型
"""


class MigrationError(Exception):
    """Base exception for migration runtime errors."""
    pass


class MigrationPathNotFoundError(MigrationError):
    """Raised when no migration path exists between source and target versions."""
    pass


class MigrationExecutionError(MigrationError):
    """Raised when an upcaster fails during migration execution."""
    pass


class SnapshotVersionTooNewError(MigrationError):
    """Snapshot version is newer than the current runtime supports."""
    pass