# src/writing/snapshot/runtime/chunking/manifest.py
"""
B3.3: StreamingManifest — 流式快照元数据
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from ..id import SnapshotId
from ..version import SemanticVersion
from .layout import ChunkLayout


@dataclass(frozen=True)
class StreamingManifest:
    """流式快照的布局与元数据（版本化）。"""

    manifest_version: SemanticVersion = field(default_factory=lambda: SemanticVersion(1, 0))
    snapshot_id: SnapshotId = ...  # type: ignore
    total_chunks: int = 0
    total_size: int = 0
    compressed_size: int = 0
    serializer_id: str = ""        # 由 Builder 注入，无默认值
    codec_id: str = ""             # 由 Builder 注入，无默认值
    layout: ChunkLayout = ...      # type: ignore
    reserved: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "reserved", MappingProxyType(dict(self.reserved)))

    def to_mapping(self) -> dict[str, Any]:
        """序列化为 JSON 兼容的 dict。"""
        return {
            "manifest_version": str(self.manifest_version),
            "snapshot_id": str(self.snapshot_id),
            "total_chunks": self.total_chunks,
            "total_size": self.total_size,
            "compressed_size": self.compressed_size,
            "serializer_id": self.serializer_id,
            "codec_id": self.codec_id,
            "layout": self.layout.to_mapping(),
            "reserved": dict(self.reserved),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "StreamingManifest":
        """从 JSON 反序列化。"""
        return cls(
            manifest_version=SemanticVersion.parse(data.get("manifest_version", "1.0")),
            snapshot_id=SnapshotId.from_string(data["snapshot_id"]),
            total_chunks=data.get("total_chunks", 0),
            total_size=data.get("total_size", 0),
            compressed_size=data.get("compressed_size", 0),
            serializer_id=data.get("serializer_id", ""),
            codec_id=data.get("codec_id", ""),
            layout=ChunkLayout.from_mapping(data.get("layout", {})),
            reserved=data.get("reserved", {}),
        )


class ManifestBuilder:
    """
    构建 Manifest（纯统计，无业务逻辑）。
    """

    def __init__(
        self,
        snapshot_id: SnapshotId,
        serializer_id: str,
        codec_id: str,
        layout: ChunkLayout,
        manifest_version: SemanticVersion | None = None,
    ):
        self._snapshot_id = snapshot_id
        self._serializer_id = serializer_id
        self._codec_id = codec_id
        self._layout = layout
        self._manifest_version = manifest_version or SemanticVersion(1, 0)
        self._chunk_count = 0
        self._compressed_size = 0
        self._reserved: dict[str, Any] = {}

    def record_chunk(self, size: int) -> None:
        """记录一个 Chunk 的大小。"""
        self._chunk_count += 1
        self._compressed_size += size

    def put_reserved(self, key: str, value: Any) -> "ManifestBuilder":
        """添加预留字段（链式调用）。"""
        self._reserved[key] = value
        return self

    def build(self, total_size: int) -> StreamingManifest:
        """完成构建，返回 Manifest。"""
        return StreamingManifest(
            manifest_version=self._manifest_version,
            snapshot_id=self._snapshot_id,
            total_chunks=self._chunk_count,
            total_size=total_size,
            compressed_size=self._compressed_size,
            serializer_id=self._serializer_id,
            codec_id=self._codec_id,
            layout=self._layout,
            reserved=self._reserved,
        )