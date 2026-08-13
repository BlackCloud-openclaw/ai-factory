# src/writing/snapshot/runtime/incremental/__init__.py
"""
B3.4: Incremental Snapshot 模块
"""

from .chunk_set import ChunkSet, EMPTY_CHUNK_SET
from .delta_chunk_set import DeltaChunkSet
from .delta_calculator import DeltaCalculator

from .version_manifest import VersionManifest, Metadata
from .version_chain import VersionChain
from .version_errors import (
    VersionError,
    VersionNotFoundError,
    VersionCycleError,
    VersionChainTooDeepError,
)
from .version_store import VersionStore, MemoryVersionStore
from .version_resolver import VersionResolver
from .chunk_repository import ChunkRepository
from .memory_chunk_repository import MemoryChunkRepository
from .incremental_transport import IncrementalTransport

__all__ = [
    # 第一阶段
    "ChunkSet",
    "EMPTY_CHUNK_SET",
    "DeltaChunkSet",
    "DeltaCalculator",
    # 第二阶段
    "VersionManifest",
    "Metadata",
    "VersionChain",
    "VersionError",
    "VersionNotFoundError",
    "VersionCycleError",
    "VersionChainTooDeepError",
    "VersionStore",
    "MemoryVersionStore",
    "VersionResolver",
    # 新增
    "ChunkRepository",
    "MemoryChunkRepository",
    "IncrementalTransport",
]