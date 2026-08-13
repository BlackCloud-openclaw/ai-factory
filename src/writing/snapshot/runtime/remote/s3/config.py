# src/writing/snapshot/runtime/remote/s3/config.py
"""
B4.3: S3 配置
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class S3Config:
    """S3 连接配置。"""

    bucket: str
    prefix: str = "snapshots/"
    region: Optional[str] = None
    endpoint_url: Optional[str] = None
    access_key: Optional[str] = None
    secret_key: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.bucket:
            raise ValueError("bucket is required")