# src/writing/snapshot/runtime/remote/gc/deletion_marker_store.py

import json
from typing import Optional, Protocol
from datetime import datetime, timedelta, timezone

from ...id import SnapshotId
from ...chunk_ref import ChunkRef
from ..s3.client import S3Client
from ..s3.key_layout import S3KeyLayout
from .errors import DeletionMarkerError, GracePeriodNotElapsedError


class DeletionMarkerStore(Protocol):
    def mark_for_deletion(self, chunk_ref: ChunkRef, grace_period_seconds: int) -> None: ...
    def get_deletion_info(self, chunk_ref: ChunkRef) -> Optional[dict]: ...
    def is_ready_for_physical_deletion(self, chunk_ref: ChunkRef) -> bool: ...
    def clear_marker(self, chunk_ref: ChunkRef) -> None: ...


class S3DeletionMarkerStore:
    def __init__(self, client: S3Client, key_layout: S3KeyLayout):
        self._client = client
        self._key_layout = key_layout

    def _marker_key(self, chunk_ref: ChunkRef) -> str:
        return f"{self._key_layout.snapshot_root_prefix()}gc-markers/{chunk_ref.snapshot_id.value}/{chunk_ref.chunk_id:08d}.json"

    def mark_for_deletion(self, chunk_ref: ChunkRef, grace_period_seconds: int) -> None:
        key = self._marker_key(chunk_ref)
        data = json.dumps({
            "delete_time": datetime.now(timezone.utc).isoformat(),
            "grace_period_seconds": grace_period_seconds,
        }).encode("utf-8")
        self._client.put_if_absent(key, data, content_type="application/json")

    def get_deletion_info(self, chunk_ref: ChunkRef) -> Optional[dict]:
        key = self._marker_key(chunk_ref)
        try:
            data = self._client.get_object(key)
            return json.loads(data.decode("utf-8"))
        except Exception:
            return None

    def is_ready_for_physical_deletion(self, chunk_ref: ChunkRef) -> bool:
        info = self.get_deletion_info(chunk_ref)
        if not info:
            raise GracePeriodNotElapsedError(f"No deletion marker for {chunk_ref}")
        delete_time = datetime.fromisoformat(info["delete_time"])
        grace = info.get("grace_period_seconds", 0)
        if datetime.now(timezone.utc) < delete_time + timedelta(seconds=grace):
            raise GracePeriodNotElapsedError(
                f"Grace period not elapsed for {chunk_ref} (deleted at {delete_time}, grace {grace}s)"
            )
        return True

    def clear_marker(self, chunk_ref: ChunkRef) -> None:
        key = self._marker_key(chunk_ref)
        self._client.delete_object(key)