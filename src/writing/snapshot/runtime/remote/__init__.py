# src/writing/snapshot/runtime/remote/__init__.py
"""
B4: Remote Storage 模块
"""

from .errors import (
    RemoteStoreError,
    ConcurrentModificationError,
    RemoteConnectionError,
    RemoteTimeoutError,
    SnapshotHasChildrenError,
)
from .repository import RemoteChunkRepository
from .cache import CachedChunkRepository
from .optimistic import OptimisticChunkRepository
from .factory import (
    create_remote_transport,
    create_s3_transport_from_env,
    create_default_registry,
)

from ..chunk_store import ChunkStore
from ..incremental import VersionStore

__all__ = [
    "RemoteStoreError",
    "ConcurrentModificationError",
    "RemoteConnectionError",
    "RemoteTimeoutError",
    "SnapshotHasChildrenError",
    "RemoteChunkRepository",
    "CachedChunkRepository",
    "OptimisticChunkRepository",
    "create_remote_transport",
    "create_s3_transport_from_env",
    "create_default_registry",
    "ChunkStore",
    "VersionStore",
]