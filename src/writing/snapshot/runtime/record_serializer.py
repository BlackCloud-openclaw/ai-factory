# src/writing/snapshot/runtime/record_serializer.py
"""
B3.1: SnapshotRecord 序列化/反序列化

格式:
    Magic(4) + Version(1) + Reserved(3) + MetadataLen(4) + Metadata + Payload
"""

import json
import struct

from .constants import RUNTIME_RECORD_FORMAT_VERSION, RUNTIME_RECORD_MAGIC
from .exceptions import SnapshotSerializationError
from .metadata import SnapshotMetadata
from .record import SnapshotRecord


def serialize_record(record: SnapshotRecord) -> bytes:
    """将 SnapshotRecord 序列化为字节流。"""
    metadata_bytes = json.dumps(
        record.metadata.to_mapping(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    metadata_len = len(metadata_bytes)

    # Magic(4) + Version(1) + Reserved(3) + MetadataLen(4) + Metadata + Payload
    header = (
        RUNTIME_RECORD_MAGIC
        + bytes([RUNTIME_RECORD_FORMAT_VERSION])
        + b"\x00\x00\x00"
        + struct.pack(">I", metadata_len)
    )
    return header + metadata_bytes + record.payload


def deserialize_record(data: bytes) -> SnapshotRecord:
    """从字节流反序列化为 SnapshotRecord。"""
    if len(data) < 12:
        raise SnapshotSerializationError("Insufficient data for record header")

    magic = data[:4]
    if magic != RUNTIME_RECORD_MAGIC:
        raise SnapshotSerializationError(
            f"Invalid magic number: expected {RUNTIME_RECORD_MAGIC!r}, got {magic!r}"
        )

    version = data[4]
    if version != RUNTIME_RECORD_FORMAT_VERSION:
        raise SnapshotSerializationError(
            f"Unsupported record version: {version} "
            f"(supported: {RUNTIME_RECORD_FORMAT_VERSION})"
        )

    # 跳过 Reserved (3 bytes)
    metadata_len = struct.unpack(">I", data[8:12])[0]
    if len(data) < 12 + metadata_len:
        raise SnapshotSerializationError("Incomplete metadata")

    metadata_bytes = data[12:12 + metadata_len]
    try:
        metadata_dict = json.loads(metadata_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise SnapshotSerializationError(f"Invalid metadata JSON: {e}") from e

    metadata = SnapshotMetadata.from_mapping(metadata_dict)

    payload = data[12 + metadata_len:]
    if len(payload) != metadata.stored_size:
        raise SnapshotSerializationError(
            f"Payload size {len(payload)} does not match stored_size {metadata.stored_size}"
        )

    return SnapshotRecord(metadata=metadata, payload=payload)