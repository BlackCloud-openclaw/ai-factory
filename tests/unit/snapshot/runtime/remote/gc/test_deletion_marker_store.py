# tests/unit/snapshot/runtime/remote/gc/test_deletion_marker_store.py
"""
B4.7.1: S3DeletionMarkerStore 单元测试
"""

import json
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

from src.writing.snapshot.runtime import SnapshotId
from src.writing.snapshot.runtime.remote.gc import (
    ChunkRef,
    S3DeletionMarkerStore,
    GracePeriodNotElapsedError,
)
from src.writing.snapshot.runtime.remote.s3 import S3Client, S3KeyLayout, S3Config
from src.writing.snapshot.runtime.remote.s3.errors import S3NotFoundError


class TestS3DeletionMarkerStore:

    def setup_method(self):
        self.config = S3Config(bucket="test-bucket", prefix="snapshots/")
        self.key_layout = S3KeyLayout(prefix="snapshots/")
        self.client = Mock(spec=S3Client)
        self.store = S3DeletionMarkerStore(self.client, self.key_layout)
        self.snapshot_id = SnapshotId.new()
        self.chunk_ref = ChunkRef(self.snapshot_id, 1)
        self.grace_period = 86400

    def test_mark_for_deletion_creates_marker(self):
        """标记删除成功创建对象。"""
        self.client.put_if_absent.return_value = True
        self.store.mark_for_deletion(self.chunk_ref, self.grace_period)
        self.client.put_if_absent.assert_called_once()
        key = self.client.put_if_absent.call_args[0][0]
        assert "gc-markers/" in key
        data = json.loads(self.client.put_if_absent.call_args[0][1].decode())
        assert "delete_time" in data
        assert data["grace_period_seconds"] == self.grace_period

    def test_mark_for_deletion_idempotent(self):
        """多次标记不覆盖（put_if_absent 返回 False 但无异常）。"""
        self.client.put_if_absent.return_value = False
        self.store.mark_for_deletion(self.chunk_ref, self.grace_period)
        self.client.put_if_absent.assert_called_once()

    def test_get_deletion_info_returns_none_when_missing(self):
        """未标记时返回 None。"""
        self.client.get_object.side_effect = S3NotFoundError("Not found")
        info = self.store.get_deletion_info(self.chunk_ref)
        assert info is None

    def test_get_deletion_info_returns_info_when_exists(self):
        """已标记返回信息。"""
        delete_time = datetime.now(timezone.utc).isoformat()
        data = json.dumps({
            "delete_time": delete_time,
            "grace_period_seconds": self.grace_period,
        }).encode()
        self.client.get_object.return_value = data
        info = self.store.get_deletion_info(self.chunk_ref)
        assert info["delete_time"] == delete_time
        assert info["grace_period_seconds"] == self.grace_period

    def test_is_ready_for_physical_deletion_raises_when_not_ready(self):
        """Grace period 未过时抛出异常。"""
        delete_time = (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat()
        data = json.dumps({
            "delete_time": delete_time,
            "grace_period_seconds": self.grace_period,  # 86400
        }).encode()
        self.client.get_object.return_value = data
        with pytest.raises(GracePeriodNotElapsedError):
            self.store.is_ready_for_physical_deletion(self.chunk_ref)

    def test_is_ready_for_physical_deletion_returns_true_when_ready(self):
        """Grace period 已过时返回 True。"""
        delete_time = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        data = json.dumps({
            "delete_time": delete_time,
            "grace_period_seconds": self.grace_period,
        }).encode()
        self.client.get_object.return_value = data
        result = self.store.is_ready_for_physical_deletion(self.chunk_ref)
        assert result is True

    def test_clear_marker_deletes_object(self):
        """清除标记删除对象。"""
        self.client.delete_object.return_value = None
        self.store.clear_marker(self.chunk_ref)
        self.client.delete_object.assert_called_once()
        key = self.client.delete_object.call_args[0][0]
        assert "gc-markers/" in key