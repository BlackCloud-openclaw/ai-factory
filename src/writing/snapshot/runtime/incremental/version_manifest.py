# src/writing/snapshot/runtime/incremental/version_manifest.py
"""
B3.4: VersionManifest — 单版本元数据
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from ..id import SnapshotId

Metadata = Mapping[str, Any]


@dataclass(frozen=True)
class VersionManifest:
    """单版本的元数据，仅包含版本关系信息和扩展元数据。"""

    snapshot_id: SnapshotId
    parent_id: SnapshotId | None = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))