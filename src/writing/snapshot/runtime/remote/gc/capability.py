# src/writing/snapshot/runtime/remote/gc/capability.py

from typing import Iterable, Protocol, runtime_checkable

from ...id import SnapshotId
from ...chunk_ref import ChunkRef
from .models import ChunkMetadata


@runtime_checkable
class ChunkEnumerator(Protocol):
    def list_all_chunks(self) -> Iterable[ChunkRef]: ...
    def list_chunks(self, snapshot_id: SnapshotId) -> Iterable[ChunkRef]: ...


@runtime_checkable
class ChunkExistenceChecker(Protocol):
    """
    B4.9 扩展点：检查单个 Chunk 是否存在。

    语义：
        - 返回 True：Chunk 确定存在。
        - 返回 False：Chunk 确定不存在（如 404）。
        - 抛出异常：检查失败（网络超时、权限不足、服务不可用等），
          调用者应决定是否终止或重试，不应将异常解释为"不存在"。
    """

    def exists(self, ref: ChunkRef) -> bool:
        """
        检查 Chunk 是否物理存在。

        Returns:
            True 如果确定存在，False 如果确定不存在。

        Raises:
            Exception: 检查失败（非 NotFound 错误）。
        """
        ...


@runtime_checkable
class ChunkMetadataProvider(Protocol):
    def get_metadata(self, chunk_ref: ChunkRef) -> ChunkMetadata: ...


@runtime_checkable
class GCDeleteAdapter(Protocol):
    def delete_chunks(self, chunks: Iterable[ChunkRef]) -> None: ...