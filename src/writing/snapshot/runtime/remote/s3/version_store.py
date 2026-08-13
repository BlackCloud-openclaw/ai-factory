# src/writing/snapshot/runtime/remote/s3/version_store.py
"""
B4.3: S3VersionStore — 实现 VersionStore Protocol
"""

import json
from typing import Iterable

from ...id import SnapshotId
from ...incremental import VersionManifest, VersionStore, VersionNotFoundError
from .client import S3Client
from .key_layout import S3KeyLayout
from .errors import S3NotFoundError, S3ConflictError


class S3VersionStore(VersionStore):
    """
    S3 实现的 VersionStore。

    仅存储 manifest.json，不关心 Chunk。
    Chunk 与 Manifest 的一致性由 RemoteChunkRepository 保证。
    """

    def __init__(
        self,
        client: S3Client,
        key_layout: S3KeyLayout | None = None,
    ):
        self._client = client
        self._key_layout = key_layout or S3KeyLayout()

    def _manifest_to_json(self, manifest: VersionManifest) -> bytes:
        """将 VersionManifest 序列化为 JSON bytes。"""
        data = {
            "snapshot_id": str(manifest.snapshot_id),
            "parent_id": str(manifest.parent_id) if manifest.parent_id else None,
            "metadata": dict(manifest.metadata),
        }
        return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def _json_to_manifest(self, data: bytes) -> VersionManifest:
        """从 JSON bytes 重建 VersionManifest。"""
        parsed = json.loads(data.decode("utf-8"))
        return VersionManifest(
            snapshot_id=SnapshotId.from_string(parsed["snapshot_id"]),
            parent_id=SnapshotId.from_string(parsed["parent_id"]) if parsed.get("parent_id") else None,
            metadata=parsed.get("metadata", {}),
        )

    def put(self, manifest: VersionManifest) -> None:
        """存储 Manifest（覆盖）。"""
        key = self._key_layout.manifest_key(manifest.snapshot_id)
        data = self._manifest_to_json(manifest)
        self._client.put_object(key, data, content_type="application/json")

    def put_if_not_exists(self, manifest: VersionManifest) -> bool:
        """原子创建 Manifest（如果不存在）。"""
        key = self._key_layout.manifest_key(manifest.snapshot_id)
        data = self._manifest_to_json(manifest)
        return self._client.put_if_absent(key, data, content_type="application/json")

    def get(self, snapshot_id: SnapshotId) -> VersionManifest:
        """读取 Manifest。"""
        key = self._key_layout.manifest_key(snapshot_id)
        try:
            data = self._client.get_object(key)
            return self._json_to_manifest(data)
        except S3NotFoundError:
            raise VersionNotFoundError(f"Version not found: {snapshot_id}")

    def delete(self, snapshot_id: SnapshotId) -> None:
        """删除 Manifest。"""
        key = self._key_layout.manifest_key(snapshot_id)
        self._client.delete_object(key)

    def list_ids(self) -> Iterable[SnapshotId]:
        """列出所有 snapshot_id（通过 manifest.json 文件）。"""
        prefix = self._key_layout._prefix
        keys = self._client.list_objects(prefix)
        ids = []
        for key in keys:
            if key.endswith("/manifest.json"):
                parts = key.split("/")
                if len(parts) >= 2:
                    snapshot_id_str = parts[-2]
                    try:
                        ids.append(SnapshotId.from_string(snapshot_id_str))
                    except ValueError:
                        continue
        return ids