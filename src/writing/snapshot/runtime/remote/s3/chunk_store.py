# src/writing/snapshot/runtime/remote/s3/chunk_store.py
"""
B4.3: S3ChunkStore — 实现 ChunkStore Protocol

ADR-B4.3-001: Manifest is the commit point.

Snapshot visibility is controlled solely by the existence of manifest.json.
Chunk uploads are non-authoritative; unreferenced chunks are orphans.
Orphan cleanup is the responsibility of future Garbage Collection (B4.5+).
"""

import json
from typing import Iterable

from ...id import SnapshotId
from ...chunk_store import ChunkStore
from ...chunking import Chunk
from .client import S3Client
from .key_layout import S3KeyLayout
from .errors import S3NotFoundError


class S3ChunkStore(ChunkStore):
    """
    S3 实现的 ChunkStore。

    使用 S3 存储单个 Chunk 和 Manifest。
    符合 B3 ChunkStore Protocol。
    """

    def __init__(
        self,
        client: S3Client,
        key_layout: S3KeyLayout | None = None,
    ):
        self._client = client
        self._key_layout = key_layout or S3KeyLayout()

    def write_chunk(self, snapshot_id: SnapshotId, chunk: Chunk) -> None:
        """存储 Chunk（幂等）。"""
        key = self._key_layout.chunk_key(snapshot_id, chunk.chunk_id)
        self._client.put_object(key, chunk.payload)

    def read_chunk(self, snapshot_id: SnapshotId, chunk_id: int) -> Chunk:
        """读取 Chunk。"""
        key = self._key_layout.chunk_key(snapshot_id, chunk_id)
        try:
            data = self._client.get_object(key)
            return Chunk(chunk_id=chunk_id, payload=data)
        except S3NotFoundError:
            raise ValueError(f"Chunk {chunk_id} not found for {snapshot_id}")

    def list_chunks(self, snapshot_id: SnapshotId) -> Iterable[int]:
        """列出所有 chunk_id。"""
        prefix = self._key_layout.list_chunks_prefix(snapshot_id)
        keys = self._client.list_objects(prefix)
        chunk_ids = []
        for key in keys:
            # 解析 chunk_id: .../chunks/00000001.bin
            filename = key.split("/")[-1]
            if filename.endswith(".bin"):
                try:
                    chunk_id = int(filename[:-4])
                    chunk_ids.append(chunk_id)
                except ValueError:
                    continue
        return sorted(chunk_ids)

    def delete(self, snapshot_id: SnapshotId) -> None:
        """
        删除所有 Chunk（不碰 manifest）。
        符合 ChunkStore Protocol 定义。
        安全删除由 RemoteChunkRepository 协调。
        """
        prefix = self._key_layout.list_chunks_prefix(snapshot_id)
        keys = self._client.list_objects(prefix)
        if keys:
            self._client.delete_objects(keys)