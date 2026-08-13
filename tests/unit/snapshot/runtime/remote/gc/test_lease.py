# tests/unit/snapshot/runtime/remote/gc/test_lease.py
"""
B4.7.1: S3LeaseManager 单元测试
"""

import json
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch, ANY

from src.writing.snapshot.runtime import SnapshotId
from src.writing.snapshot.runtime.remote.gc import ChunkRef, S3LeaseManager
from src.writing.snapshot.runtime.remote.gc.errors import (
    LeaseAcquisitionError,
    LeaseRenewalError,
    LeaseReleaseError,
    LeaseConflictError,
)
from src.writing.snapshot.runtime.remote.s3 import S3Client, S3KeyLayout, S3Config
from src.writing.snapshot.runtime.remote.s3.errors import S3NotFoundError, S3ConflictError


class TestS3LeaseManager:

    def setup_method(self):
        self.config = S3Config(bucket="test-bucket", prefix="snapshots/")
        self.key_layout = S3KeyLayout(prefix="snapshots/")
        self.client = Mock(spec=S3Client)
        self.manager = S3LeaseManager(self.client, self.key_layout)
        self.snapshot_id = SnapshotId.new()
        self.chunk_ref = ChunkRef(self.snapshot_id, 1)
        self.owner_id = "gc-host-123"

    def test_acquire_creates_new_lease(self):
        """成功创建新租约（put_if_absent 返回 True）。"""
        self.client.put_if_absent.return_value = True
        result = self.manager.acquire(self.chunk_ref, 60, self.owner_id)
        assert result is True
        self.client.put_if_absent.assert_called_once_with(
            ANY, ANY, content_type="application/json"
        )
        # 验证 key 包含 leases/ 和 chunk id
        key = self.client.put_if_absent.call_args[0][0]
        assert "leases/" in key
        assert f"/{self.chunk_ref.chunk_id:08d}.json" in key

    def test_acquire_when_lease_exists_and_not_expired(self):
        """租约已存在且未过期，且不属于自己，抛出 LeaseConflictError。"""
        expires = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
        existing_data = json.dumps({"owner": "other-node", "expires": expires}).encode()
        self.client.put_if_absent.return_value = False
        self.client.head_object.return_value = {"ETag": '"abc123"'}
        self.client.get_object.return_value = existing_data

        with pytest.raises(LeaseConflictError) as exc:
            self.manager.acquire(self.chunk_ref, 60, self.owner_id)
        assert "held by other-node" in str(exc.value)

    def test_acquire_when_lease_exists_and_expired(self):
        """租约已过期，CAS 替换成功，返回 True。"""
        expires = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        existing_data = json.dumps({"owner": "old-node", "expires": expires}).encode()
        self.client.put_if_absent.return_value = False
        self.client.head_object.return_value = {"ETag": '"abc123"'}
        self.client.get_object.return_value = existing_data
        self.client.replace_if_match.return_value = True

        result = self.manager.acquire(self.chunk_ref, 60, self.owner_id)
        assert result is True
        self.client.replace_if_match.assert_called_once()

    def test_acquire_when_lease_exists_and_expired_but_cas_fails(self):
        """租约已过期，但 CAS 替换失败（竞争），返回 False。"""
        expires = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        existing_data = json.dumps({"owner": "old-node", "expires": expires}).encode()
        self.client.put_if_absent.return_value = False
        self.client.head_object.return_value = {"ETag": '"abc123"'}
        self.client.get_object.return_value = existing_data
        self.client.replace_if_match.return_value = False

        result = self.manager.acquire(self.chunk_ref, 60, self.owner_id)
        assert result is False

    def test_acquire_raises_on_system_error(self):
        """网络错误等导致 LeaseAcquisitionError。"""
        self.client.put_if_absent.side_effect = Exception("Network timeout")
        with pytest.raises(LeaseAcquisitionError):
            self.manager.acquire(self.chunk_ref, 60, self.owner_id)

    def test_renew_success(self):
        """续租成功。"""
        expires = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
        existing_data = json.dumps({"owner": self.owner_id, "expires": expires}).encode()
        self.client.head_object.return_value = {"ETag": '"abc123"'}
        self.client.get_object.return_value = existing_data
        self.client.replace_if_match.return_value = True

        result = self.manager.renew(self.chunk_ref, 60, self.owner_id)
        assert result is True
        self.client.replace_if_match.assert_called_once()

    def test_renew_fails_when_owner_mismatch(self):
        """owner 不匹配返回 False。"""
        expires = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
        existing_data = json.dumps({"owner": "other-node", "expires": expires}).encode()
        self.client.head_object.return_value = {"ETag": '"abc123"'}
        self.client.get_object.return_value = existing_data

        result = self.manager.renew(self.chunk_ref, 60, self.owner_id)
        assert result is False

    def test_renew_fails_when_expired(self):
        """租约已过期返回 False。"""
        expires = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        existing_data = json.dumps({"owner": self.owner_id, "expires": expires}).encode()
        self.client.head_object.return_value = {"ETag": '"abc123"'}
        self.client.get_object.return_value = existing_data

        result = self.manager.renew(self.chunk_ref, 60, self.owner_id)
        assert result is False

    def test_renew_raises_on_system_error(self):
        """续租时系统错误抛出 LeaseRenewalError。"""
        self.client.head_object.side_effect = Exception("S3 unavailable")
        with pytest.raises(LeaseRenewalError):
            self.manager.renew(self.chunk_ref, 60, self.owner_id)

    def test_release_success(self):
        """释放租约（owner 匹配）。"""
        expires = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
        existing_data = json.dumps({"owner": self.owner_id, "expires": expires}).encode()
        self.client.get_object.return_value = existing_data

        self.manager.release(self.chunk_ref, self.owner_id)
        self.client.delete_object.assert_called_once()

    def test_release_skip_when_owner_mismatch(self):
        """owner 不匹配时不删除。"""
        expires = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
        existing_data = json.dumps({"owner": "other-node", "expires": expires}).encode()
        self.client.get_object.return_value = existing_data

        self.manager.release(self.chunk_ref, self.owner_id)
        self.client.delete_object.assert_not_called()

    def test_release_raises_on_error(self):
        """释放时系统错误抛出 LeaseReleaseError。"""
        self.client.get_object.side_effect = Exception("S3 error")
        with pytest.raises(LeaseReleaseError):
            self.manager.release(self.chunk_ref, self.owner_id)

    def test_is_held_returns_true_when_valid(self):
        """租约有效返回 True。"""
        expires = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
        existing_data = json.dumps({"owner": self.owner_id, "expires": expires}).encode()
        self.client.get_object.return_value = existing_data

        assert self.manager.is_held(self.chunk_ref) is True

    def test_is_held_returns_false_when_expired(self):
        """租约过期返回 False。"""
        expires = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        existing_data = json.dumps({"owner": self.owner_id, "expires": expires}).encode()
        self.client.get_object.return_value = existing_data

        assert self.manager.is_held(self.chunk_ref) is False

    def test_is_held_returns_false_on_error(self):
        """获取租约失败返回 False（对象不存在等）。"""
        self.client.get_object.side_effect = S3NotFoundError("Not found")
        assert self.manager.is_held(self.chunk_ref) is False