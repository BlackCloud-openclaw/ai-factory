# src/writing/snapshot/runtime/incremental/version_resolver.py
"""
B3.4: VersionResolver — 版本链解析（使用 ChunkRepository）
"""

from ..id import SnapshotId
from .chunk_repository import ChunkRepository
from .version_chain import VersionChain
from .version_errors import (
    VersionNotFoundError,
    VersionCycleError,
    VersionChainTooDeepError,
)


class VersionResolver:
    """
    版本链解析器。

    从任意 SnapshotId 追溯到 Base，返回 VersionChain。
    内置循环检测和深度限制。
    依赖 ChunkRepository 获取 VersionManifest。
    """

    def __init__(self, repository: ChunkRepository, max_depth: int = 32):
        self._repository = repository
        self._max_depth = max_depth

    def resolve_chain(self, snapshot_id: SnapshotId) -> VersionChain:
        """从 snapshot_id 追溯到 Base，返回 Base -> Latest 顺序的 VersionChain。"""
        current = snapshot_id
        visited: set[SnapshotId] = set()
        chain: list[SnapshotId] = []

        while True:
            if current in visited:
                cycle = " -> ".join(str(v) for v in chain + [current])
                raise VersionCycleError(f"Cycle detected: {cycle}")

            visited.add(current)

            try:
                manifest = self._repository.load_manifest(current)
            except VersionNotFoundError as e:
                raise VersionNotFoundError(
                    f"Version {current} referenced but not found in repository"
                ) from e

            chain.append(current)

            if len(chain) > self._max_depth:
                raise VersionChainTooDeepError(
                    f"Chain depth {len(chain)} exceeds max_depth {self._max_depth}"
                )

            if manifest.parent_id is None:
                break

            current = manifest.parent_id

        chain.reverse()
        return VersionChain(tuple(chain))