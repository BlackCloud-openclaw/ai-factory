# src/writing/snapshot/runtime/metadata.py
"""
B3.1/B3.2: SnapshotMetadata — Runtime Record 元数据
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from .constants import RUNTIME_RECORD_FORMAT_VERSION


def _validate_json_compatible(value: Any) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise TypeError(f"dict keys must be str, got {type(k).__name__}")
            _validate_json_compatible(v)
    elif isinstance(value, list):
        for v in value:
            _validate_json_compatible(v)
    elif isinstance(value, (str, int, float, bool, type(None))):
        pass
    else:
        raise TypeError(f"Value {type(value).__name__} is not JSON-compatible")


@dataclass(frozen=True)
class SnapshotMetadata:
    format_version: int = RUNTIME_RECORD_FORMAT_VERSION
    serializer: str = "builtin.json"
    codec_id: str = "builtin.identity"
    content_size: int = 0
    stored_size: int = 0
    reserved: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_json_compatible(self.reserved)
        object.__setattr__(self, "reserved", MappingProxyType(dict(self.reserved)))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "serializer": self.serializer,
            "codec_id": self.codec_id,
            "content_size": self.content_size,
            "stored_size": self.stored_size,
            "reserved": dict(self.reserved),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SnapshotMetadata":
        return cls(
            format_version=data.get("format_version", RUNTIME_RECORD_FORMAT_VERSION),
            serializer=data.get("serializer", "builtin.json"),
            codec_id=data.get("codec_id", "builtin.identity"),
            content_size=data.get("content_size", 0),
            stored_size=data.get("stored_size", 0),
            reserved=data.get("reserved", {}),
        )