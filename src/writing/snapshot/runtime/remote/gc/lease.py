# src/writing/snapshot/runtime/remote/gc/lease.py
"""
B4.7/B4.11: Lease Manager — 租约管理（Chunk 级别 + Scope 级别）
"""

import json
from typing import Protocol, Optional, ContextManager
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager

from ...id import SnapshotId
from ...chunk_ref import ChunkRef
from ..s3.client import S3Client
from ..s3.key_layout import S3KeyLayout
from .errors import (
    LeaseAcquisitionError,
    LeaseRenewalError,
    LeaseReleaseError,
    LeaseConflictError,
    GarbageCollectionError,
)


class LeaseManager(Protocol):
    """租约管理协议（Chunk 级别 + Scope 级别）。"""

    # Chunk 级别租约
    def acquire(self, chunk_ref: ChunkRef, ttl_seconds: int, owner_id: str) -> bool:
        ...
    def renew(self, chunk_ref: ChunkRef, ttl_seconds: int, owner_id: str) -> bool:
        ...
    def release(self, chunk_ref: ChunkRef, owner_id: str) -> None:
        ...
    def is_held(self, chunk_ref: ChunkRef) -> bool:
        ...

    # Scope 级别租约（B4.11）
    def acquire_scope(self, scope_id: str, ttl_seconds: int) -> bool:
        ...
    def release_scope(self, scope_id: str) -> None:
        ...
    def is_scope_held(self, scope_id: str) -> bool:
        ...

    # 上下文管理器（B4.11）
    def scope(self, scope_id: str, ttl_seconds: int) -> ContextManager[None]:
        ...


class S3LeaseManager:
    """
    基于 S3 条件写入的安全租约实现（Chunk 级别 + Scope 级别）。
    """

    def __init__(self, client: S3Client, key_layout: S3KeyLayout):
        self._client = client
        self._key_layout = key_layout

    # ========== Chunk 级别租约（B4.7） ==========

    def _lease_key(self, chunk_ref: ChunkRef) -> str:
        return f"{self._key_layout.snapshot_root_prefix()}leases/{chunk_ref.snapshot_id.value}/chunks/{chunk_ref.chunk_id:08d}.json"

    def _encode_lease(self, owner_id: str, expires: datetime) -> bytes:
        return json.dumps({"owner": owner_id, "expires": expires.isoformat()}).encode("utf-8")

    def _decode_lease(self, data: bytes) -> dict:
        return json.loads(data.decode("utf-8"))

    def acquire(self, chunk_ref: ChunkRef, ttl_seconds: int, owner_id: str) -> bool:
        key = self._lease_key(chunk_ref)
        expires = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
        data = self._encode_lease(owner_id, expires)

        try:
            created = self._client.put_if_absent(key, data, content_type="application/json")
            if created:
                return True
        except Exception as e:
            raise LeaseAcquisitionError(f"Failed to create lease for {chunk_ref}: {e}") from e

        try:
            head = self._client.head_object(key)
            if head is None:
                return self._client.put_if_absent(key, data, content_type="application/json")

            etag = head.get("ETag", "")
            if not etag:
                raise LeaseAcquisitionError("Missing ETag for lease replacement")

            existing_data = self._client.get_object(key)
            existing = self._decode_lease(existing_data)
            existing_expires = datetime.fromisoformat(existing["expires"])
            existing_owner = existing["owner"]

            if existing_expires >= datetime.now(timezone.utc):
                if existing_owner == owner_id:
                    return True
                raise LeaseConflictError(f"Lease held by {existing_owner} until {existing_expires}")

            success = self._client.replace_if_match(key, etag, data, content_type="application/json")
            return success

        except LeaseConflictError:
            raise
        except Exception as e:
            raise LeaseAcquisitionError(f"Failed to acquire lease for {chunk_ref}: {e}") from e

    def renew(self, chunk_ref: ChunkRef, ttl_seconds: int, owner_id: str) -> bool:
        key = self._lease_key(chunk_ref)
        try:
            head = self._client.head_object(key)
            if head is None:
                return False
            etag = head.get("ETag", "")
            if not etag:
                raise LeaseRenewalError("Missing ETag for lease renewal")

            existing_data = self._client.get_object(key)
            existing = self._decode_lease(existing_data)

            if existing.get("owner") != owner_id:
                return False

            expires_str = existing.get("expires")
            if expires_str:
                exp_time = datetime.fromisoformat(expires_str)
                if exp_time < datetime.now(timezone.utc):
                    return False

            new_expires = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
            new_data = self._encode_lease(owner_id, new_expires)

            success = self._client.replace_if_match(key, etag, new_data, content_type="application/json")
            return success

        except Exception as e:
            raise LeaseRenewalError(f"Failed to renew lease for {chunk_ref}: {e}") from e

    def release(self, chunk_ref: ChunkRef, owner_id: str) -> None:
        key = self._lease_key(chunk_ref)
        try:
            existing_data = self._client.get_object(key)
            existing = self._decode_lease(existing_data)
            if existing.get("owner") != owner_id:
                return
            self._client.delete_object(key)
        except Exception as e:
            raise LeaseReleaseError(f"Failed to release lease for {chunk_ref}: {e}") from e

    def is_held(self, chunk_ref: ChunkRef) -> bool:
        key = self._lease_key(chunk_ref)
        try:
            existing_data = self._client.get_object(key)
            existing = self._decode_lease(existing_data)
            expires_str = existing.get("expires")
            if expires_str:
                exp_time = datetime.fromisoformat(expires_str)
                return exp_time >= datetime.now(timezone.utc)
            return False
        except Exception:
            return False

    # ========== Scope 级别租约（B4.11） ==========

    def _scope_lease_key(self, scope_id: str) -> str:
        return f"{self._key_layout.snapshot_root_prefix()}leases/_scopes/{scope_id}.json"

    def acquire_scope(self, scope_id: str, ttl_seconds: int) -> bool:
        key = self._scope_lease_key(scope_id)
        expires = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
        data = self._encode_lease(scope_id, expires)
        try:
            return self._client.put_if_absent(key, data, content_type="application/json")
        except Exception:
            return False

    def release_scope(self, scope_id: str) -> None:
        key = self._scope_lease_key(scope_id)
        try:
            self._client.delete_object(key)
        except Exception:
            pass

    def is_scope_held(self, scope_id: str) -> bool:
        key = self._scope_lease_key(scope_id)
        try:
            data = self._client.get_object(key)
            info = self._decode_lease(data)
            exp = datetime.fromisoformat(info["expires"])
            return exp > datetime.now(timezone.utc)
        except Exception:
            return False

    def scope(self, scope_id: str, ttl_seconds: int) -> ContextManager[None]:
        """返回作用域租约上下文管理器。"""
        return _LeaseScope(self, scope_id, ttl_seconds)


class _LeaseScope:
    """作用域租约上下文管理器（内部类）。"""
    def __init__(self, lease_manager: S3LeaseManager, scope_id: str, ttl_seconds: int):
        self._lease_manager = lease_manager
        self._scope_id = scope_id
        self._ttl_seconds = ttl_seconds

    def __enter__(self):
        if not self._lease_manager.acquire_scope(self._scope_id, self._ttl_seconds):
            raise GarbageCollectionError(f"Failed to acquire scope lease: {self._scope_id}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lease_manager.release_scope(self._scope_id)