# src/writing/snapshot/runtime/incremental/version_chain.py
"""
B3.4: VersionChain — 版本链值对象
"""

from dataclasses import dataclass
from typing import Iterator

from ..id import SnapshotId


@dataclass(frozen=True)
class VersionChain:
    """从 Base 到 Latest 的版本列表（值对象）。"""

    versions: tuple[SnapshotId, ...]

    def __post_init__(self) -> None:
        if not self.versions:
            raise ValueError("VersionChain cannot be empty")

    @property
    def base(self) -> SnapshotId:
        return self.versions[0]

    @property
    def latest(self) -> SnapshotId:
        return self.versions[-1]

    @property
    def depth(self) -> int:
        return len(self.versions)

    def __len__(self) -> int:
        return len(self.versions)

    def __iter__(self) -> Iterator[SnapshotId]:
        return iter(self.versions)

    def __getitem__(self, idx: int) -> SnapshotId:
        return self.versions[idx]

    def __contains__(self, item: object) -> bool:
        return isinstance(item, SnapshotId) and item in self.versions