# src/writing/snapshot/runtime/record.py
"""
B3.1: SnapshotRecord — Runtime 存储记录
"""

from dataclasses import dataclass

from .metadata import SnapshotMetadata


@dataclass(frozen=True)
class SnapshotRecord:
    """存储层统一记录，包含元数据和字节 payload。"""

    metadata: SnapshotMetadata
    payload: bytes

    def __post_init__(self) -> None:
        """校验 payload 大小与 metadata.stored_size 一致。"""
        if len(self.payload) != self.metadata.stored_size:
            raise ValueError(
                f"Payload size {len(self.payload)} does not match "
                f"metadata.stored_size {self.metadata.stored_size}"
            )