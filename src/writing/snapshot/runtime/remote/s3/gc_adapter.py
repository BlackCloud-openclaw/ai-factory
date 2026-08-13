# src/writing/snapshot/runtime/remote/s3/gc_adapter.py

from typing import Iterable, Optional
from datetime import datetime

from ...id import SnapshotId
from ..gc import ChunkRef, ChunkMetadata, ChunkEnumerator, GCDeleteAdapter, ChunkMetadataProvider
from .client import S3Client
from .key_layout import S3KeyLayout
from .errors import S3NotFoundError


class S3GCAdapter(ChunkEnumerator, GCDeleteAdapter, ChunkMetadataProvider):
    """
    S3 后端的 GC 能力适配器。

    实现：
        - ChunkEnumerator: 枚举物理 Chunk（全量 + 按 Snapshot）
        - GCDeleteAdapter: 批量删除 Chunk
        - ChunkMetadataProvider: 获取 Chunk 物理元数据
    """

    DELETE_BATCH_SIZE = 1000

    def __init__(
        self,
        client: S3Client,
        key_layout: S3KeyLayout,
    ):
        self._client = client
        self._layout = key_layout

    # ========== ChunkEnumerator 实现 ==========

    def list_all_chunks(self) -> Iterable[ChunkRef]:
        prefix = self._layout.snapshot_root_prefix()
        for key in self._client.list_objects(prefix):
            ref = self._layout.parse_chunk_key(key)
            if ref is not None:
                yield ref

    def list_chunks(self, snapshot_id: SnapshotId) -> Iterable[ChunkRef]:
        prefix = self._layout.list_chunks_prefix(snapshot_id)
        for key in self._client.list_objects(prefix):
            ref = self._layout.parse_chunk_key(key)
            if ref is not None and ref.snapshot_id == snapshot_id:
                yield ref

    # ========== ChunkMetadataProvider 实现 ==========

    def get_metadata(self, chunk_ref: ChunkRef) -> ChunkMetadata:
        """
        通过 S3 head_object 获取 Chunk 元数据。

        映射：
            ContentLength → size_bytes
            ETag → checksum
            LastModified → created_at
        """
        key = self._layout.chunk_key(chunk_ref.snapshot_id, chunk_ref.chunk_id)
        try:
            response = self._client.head_object(key)
        except S3NotFoundError:
            raise ValueError(f"Chunk not found: {chunk_ref}")

        # 提取元数据
        size_bytes = response.get("ContentLength", 0)
        checksum = response.get("ETag")  # ETag 已包含引号，保留原样
        last_modified = response.get("LastModified")
        created_at = None
        if last_modified:
            # 如果 LastModified 是字符串，尝试解析
            if isinstance(last_modified, str):
                try:
                    # ISO 8601 格式
                    created_at = datetime.fromisoformat(last_modified.replace("Z", "+00:00"))
                except ValueError:
                    pass
            else:
                created_at = last_modified

        return ChunkMetadata(
            chunk_ref=chunk_ref,
            size_bytes=size_bytes,
            checksum=checksum,
            created_at=created_at,
        )

    # ========== GCDeleteAdapter 实现 ==========

    def delete_chunks(self, chunks: Iterable[ChunkRef]) -> None:
        keys = []
        for ref in chunks:
            key = self._layout.chunk_key(ref.snapshot_id, ref.chunk_id)
            keys.append(key)

        if not keys:
            return

        for i in range(0, len(keys), self.DELETE_BATCH_SIZE):
            batch = keys[i:i + self.DELETE_BATCH_SIZE]
            self._client.delete_objects(batch)