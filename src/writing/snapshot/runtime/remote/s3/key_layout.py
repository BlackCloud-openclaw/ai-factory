# src/writing/snapshot/runtime/remote/s3/key_layout.py

import re
from typing import Optional

from ...id import SnapshotId
from ...chunk_ref import ChunkRef


class S3KeyLayout:
    def __init__(self, prefix: str = "snapshots/"):
        self._prefix = prefix.rstrip("/") + "/" if prefix else ""
        self._chunk_key_pattern = re.compile(
            rf"^{re.escape(self._prefix)}([^/]+)/chunks/(\d{{8}})\.bin$"
        )
        self._marker_key_pattern = re.compile(
            rf"^{re.escape(self._prefix)}gc-markers/([^/]+)/(\d{{8}})\.json$"
        )

    def manifest_key(self, snapshot_id: SnapshotId) -> str:
        return f"{self._prefix}{snapshot_id.value}/manifest.json"

    def chunk_key(self, snapshot_id: SnapshotId, chunk_id: int) -> str:
        return f"{self._prefix}{snapshot_id.value}/chunks/{chunk_id:08d}.bin"

    def snapshot_root_prefix(self) -> str:
        return self._prefix

    def snapshot_prefix(self, snapshot_id: SnapshotId) -> str:
        return f"{self._prefix}{snapshot_id.value}/"

    def list_chunks_prefix(self, snapshot_id: SnapshotId) -> str:
        return f"{self._prefix}{snapshot_id.value}/chunks/"

    def marker_prefix(self, snapshot_id: Optional[SnapshotId] = None) -> str:
        base = f"{self._prefix}gc-markers/"
        if snapshot_id is None:
            return base
        return f"{base}{snapshot_id.value}/"

    def parse_chunk_key(self, key: str) -> Optional[ChunkRef]:
        match = self._chunk_key_pattern.match(key)
        if not match:
            return None
        snapshot_id_str, chunk_id_str = match.groups()
        try:
            snapshot_id = SnapshotId.from_string(snapshot_id_str)
            chunk_id = int(chunk_id_str)
            return ChunkRef(snapshot_id, chunk_id)
        except (ValueError, TypeError):
            return None

    def parse_marker_key(self, key: str) -> Optional[ChunkRef]:
        match = self._marker_key_pattern.match(key)
        if not match:
            return None
        snapshot_id_str, chunk_id_str = match.groups()
        try:
            snapshot_id = SnapshotId.from_string(snapshot_id_str)
            chunk_id = int(chunk_id_str)
            return ChunkRef(snapshot_id, chunk_id)
        except (ValueError, TypeError):
            return None