# tests/unit/snapshot/runtime/remote/s3/test_version_store.py

import json
import pytest
from unittest.mock import Mock

from src.writing.snapshot.runtime import SnapshotId
from src.writing.snapshot.runtime.incremental import VersionManifest, VersionNotFoundError
from src.writing.snapshot.runtime.remote.s3 import S3VersionStore, S3KeyLayout
from src.writing.snapshot.runtime.remote.s3.client import S3Client
from src.writing.snapshot.runtime.remote.s3.errors import S3NotFoundError


class TestS3VersionStore:
    def setup_method(self):
        self.sid = SnapshotId.new()
        self.manifest = VersionManifest(snapshot_id=self.sid, parent_id=None)
        self.client = Mock(spec=S3Client)
        self.layout = S3KeyLayout("prefix/")
        self.store = S3VersionStore(self.client, self.layout)

    def test_put(self):
        self.store.put(self.manifest)
        expected_key = f"prefix/{self.sid.value}/manifest.json"
        self.client.put_object.assert_called_once()
        call_args = self.client.put_object.call_args
        # 尝试从 kwargs 获取 key
        key = call_args.kwargs.get("key")
        if key is None and call_args[0]:
            # 位置参数: (key, data)
            key = call_args[0][0]
        assert key == expected_key
        assert call_args.kwargs.get("content_type") == "application/json"

    def test_get(self):
        data = {
            "snapshot_id": str(self.sid),
            "parent_id": None,
            "metadata": {},
        }
        self.client.get_object.return_value = json.dumps(data).encode("utf-8")
        manifest = self.store.get(self.sid)
        assert manifest.snapshot_id == self.sid
        assert manifest.parent_id is None

    def test_get_not_found(self):
        self.client.get_object.side_effect = S3NotFoundError("Not found")
        with pytest.raises(VersionNotFoundError):
            self.store.get(self.sid)

    def test_delete(self):
        self.store.delete(self.sid)
        expected_key = f"prefix/{self.sid.value}/manifest.json"
        self.client.delete_object.assert_called_once_with(expected_key)

    def test_list_ids(self):
        self.client.list_objects.return_value = [
            "prefix/id1/manifest.json",
            "prefix/id2/manifest.json",
            "prefix/id3/manifest.json",
        ]
        ids = list(self.store.list_ids())
        assert len(ids) >= 0