# src/writing/snapshot/runtime/remote/s3/client.py
"""
B4.3: S3 客户端抽象（隔离 boto3）
"""

import json
from typing import Optional, Iterator, BinaryIO
from dataclasses import dataclass
import time

from .config import S3Config
from .errors import (
    S3Error,
    S3ConnectionError,
    S3TimeoutError,
    S3NotFoundError,
    S3ConflictError,
)

try:
    import boto3
    from botocore.exceptions import (
        ClientError,
        ConnectionError,
        ReadTimeoutError,
        ParamValidationError,
    )
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    ClientError = None
    ConnectionError = None
    ReadTimeoutError = None


@dataclass(frozen=True)
class S3ObjectSummary:
    """S3 对象摘要（类型安全）。"""
    key: str
    size: int
    etag: str


class S3Client:
    def __init__(self, config: S3Config):
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required for S3 backend")
        self._config = config
        self._client = self._create_client()

    def _create_client(self):
        kwargs = {"service_name": "s3"}
        if self._config.region:
            kwargs["region_name"] = self._config.region
        if self._config.endpoint_url:
            kwargs["endpoint_url"] = self._config.endpoint_url
        if self._config.access_key and self._config.secret_key:
            kwargs["aws_access_key_id"] = self._config.access_key
            kwargs["aws_secret_access_key"] = self._config.secret_key
        return boto3.client(**kwargs)

    def _handle_error(self, error: Exception, operation: str) -> None:
        if ClientError is None:
            raise S3Error(f"{operation} failed: {error}") from error

        if isinstance(error, ClientError):
            status_code = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
            error_code = error.response.get("Error", {}).get("Code", "")

            if status_code == 404 or error_code == "NoSuchKey":
                raise S3NotFoundError(f"Object not found in S3") from error
            if status_code == 409 or error_code in ("Conflict", "ConditionFailed"):
                raise S3ConflictError(f"Conflict in S3: {error_code}") from error

            transient_codes = {"SlowDown", "ServiceUnavailable", "InternalError"}
            if status_code >= 500 or error_code in transient_codes:
                raise S3ConnectionError(f"S3 transient error: {error_code}") from error

            raise S3Error(f"S3 error: {error_code}") from error

        if ConnectionError and isinstance(error, ConnectionError):
            raise S3ConnectionError(f"S3 connection error: {error}") from error

        if ReadTimeoutError and isinstance(error, ReadTimeoutError):
            raise S3TimeoutError(f"S3 timeout: {error}") from error

        raise S3Error(f"Unexpected S3 error: {error}") from error

    def put_object(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        try:
            response = self._client.put_object(
                Bucket=self._config.bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
            )
            return response.get("ETag", "")
        except Exception as e:
            self._handle_error(e, f"PUT {key}")

    def put_if_absent(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> bool:
        try:
            self._client.put_object(
                Bucket=self._config.bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
                IfNoneMatch="*",
            )
            return True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "PreconditionFailed":
                return False
            self._handle_error(e, f"PUT (if-absent) {key}")

    def replace_if_match(self, key: str, etag: str, data: bytes, content_type: str = "application/octet-stream") -> bool:
        try:
            self._client.put_object(
                Bucket=self._config.bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
                IfMatch=etag,
            )
            return True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "PreconditionFailed":
                return False
            self._handle_error(e, f"PUT (if-match) {key}")

    def get_object(self, key: str) -> bytes:
        try:
            response = self._client.get_object(Bucket=self._config.bucket, Key=key)
            return response["Body"].read()
        except Exception as e:
            self._handle_error(e, f"GET {key}")

    def head_object(self, key: str) -> dict:
        try:
            return self._client.head_object(Bucket=self._config.bucket, Key=key)
        except ClientError as e:
            if e.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404:
                return None
            self._handle_error(e, f"HEAD {key}")

    def delete_object(self, key: str) -> None:
        try:
            self._client.delete_object(Bucket=self._config.bucket, Key=key)
        except Exception as e:
            self._handle_error(e, f"DELETE {key}")

    def delete_objects(self, keys: list[str]) -> None:
        if not keys:
            return
        try:
            self._client.delete_objects(
                Bucket=self._config.bucket,
                Delete={"Objects": [{"Key": k} for k in keys]},
            )
        except Exception as e:
            self._handle_error(e, f"DELETE {len(keys)} objects")

    def list_objects(self, prefix: str) -> list[str]:
        keys = []
        try:
            paginator = self._client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self._config.bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    keys.append(obj["Key"])
            return keys
        except Exception as e:
            self._handle_error(e, f"LIST {prefix}")

    def upload_stream(self, key: str, stream, content_type: str = "application/octet-stream") -> str:
        raise NotImplementedError("Streaming upload will be implemented in B4.4")

    def download_stream(self, key: str):
        raise NotImplementedError("Streaming download will be implemented in B4.4")

    # ========== B4.8 流式分页迭代 ==========
    def iter_objects(self, prefix: str) -> Iterator[S3ObjectSummary]:
        try:
            paginator = self._client.get_paginator("list_objects_v2")
            for page in paginator.paginate(
                Bucket=self._config.bucket,
                Prefix=prefix,
            ):
                for obj in page.get("Contents", []):
                    yield S3ObjectSummary(
                        key=obj["Key"],
                        size=obj.get("Size", 0),
                        etag=obj.get("ETag", "").strip('"'),
                    )
        except Exception as e:
            self._handle_error(e, f"LIST (paginated) {prefix}")