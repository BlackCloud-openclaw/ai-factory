# src/writing/snapshot/migration/version.py
"""
B1.1 Version Model — 核心领域类型（API Frozen）

此版本为 Release B 的最终冻结 API，任何后续变更必须通过 ADR 评审。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Mapping,
    Protocol,
    TypeVar,
    runtime_checkable,
)

if TYPE_CHECKING:
    from .raw_snapshot import RawSnapshot  # B1.2 定义


# ============================================================================
# 泛型 Helper
# ============================================================================

K = TypeVar("K")
V = TypeVar("V")


def freeze_mapping(mapping: Mapping[K, V]) -> Mapping[K, V]:
    """将 dict 转换为不可变的 MappingProxyType。"""
    return MappingProxyType(dict(mapping))


# ============================================================================
# Clock / RandomProvider
# ============================================================================

@runtime_checkable
class Clock(Protocol):
    def now(self) -> datetime:
        ...


@runtime_checkable
class RandomProvider(Protocol):
    def generate(self) -> str:
        ...


class FixedClock(Clock):
    def __init__(self, fixed_time: datetime = datetime(1970, 1, 1, tzinfo=timezone.utc)):
        self._fixed_time = fixed_time

    def now(self) -> datetime:
        return self._fixed_time


class FixedRandom(RandomProvider):
    def __init__(self, seed: str = "fixed-seed"):
        self._seed = seed

    def generate(self) -> str:
        return self._seed


# ============================================================================
# SchemaVersion
# ============================================================================

@dataclass(frozen=True, order=True)
class SchemaVersion:
    major: int
    minor: int

    def __post_init__(self) -> None:
        if self.major < 0 or self.minor < 0:
            raise ValueError(f"Version components must be non-negative: {self.major}.{self.minor}")

    @classmethod
    def parse(cls, value: str) -> SchemaVersion:
        """严格解析，不进行 normalize。Normalization 由 Loader 负责。"""
        parts = value.strip().split(".")
        if len(parts) != 2:
            raise ValueError(f"Invalid version format: {value} (expected 'major.minor')")
        try:
            return cls(major=int(parts[0]), minor=int(parts[1]))
        except ValueError as e:
            raise ValueError(f"Invalid version components: {value}") from e

    @property
    def components(self) -> tuple[int, int]:
        return (self.major, self.minor)

    def next_minor(self) -> SchemaVersion:
        return SchemaVersion(major=self.major, minor=self.minor + 1)

    def next_major(self) -> SchemaVersion:
        return SchemaVersion(major=self.major + 1, minor=0)

    def is_major_upgrade_to(self, other: SchemaVersion) -> bool:
        return self.major < other.major

    def is_minor_upgrade_to(self, other: SchemaVersion) -> bool:
        return self.major == other.major and self.minor < other.minor

    def is_upgrade_to(self, other: SchemaVersion) -> bool:
        return self.is_major_upgrade_to(other) or self.is_minor_upgrade_to(other)

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}"

    def is_newer_than(self, other: SchemaVersion) -> bool:
        """判断当前版本是否比 other 更新（大于）。"""
        return self > other

    def is_older_than(self, other: SchemaVersion) -> bool:
        """判断当前版本是否比 other 更旧（小于）。"""
        return self < other
    
    @classmethod
    def from_string(cls, value: str) -> "SchemaVersion":
        """从字符串解析版本号（与 parse 相同，更符合命名习惯）。"""
        return cls.parse(value)

# ============================================================================
# CapabilityId
# ============================================================================

_CAPABILITY_ID_PATTERN = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)+$")


@dataclass(frozen=True)
class CapabilityId:
    """能力标识符。严格解析，不自动 normalize。"""

    value: str

    def __post_init__(self) -> None:
        if not _CAPABILITY_ID_PATTERN.match(self.value):
            raise ValueError(
                f"Invalid CapabilityId: '{self.value}'. "
                f"Must match pattern: [a-z0-9_]+(\.[a-z0-9_]+)+"
            )

    @classmethod
    def parse(cls, value: str) -> CapabilityId:
        """严格解析，不进行 normalize。"""
        return cls(value=value)

    def __str__(self) -> str:
        return self.value


# ============================================================================
# VersionType
# ============================================================================

class VersionType(Enum):
    MAJOR = "major"
    MINOR = "minor"
    # 预留: PATCH, LTS, EXPERIMENTAL


# ============================================================================
# VersionNode
# ============================================================================

@dataclass(frozen=True)
class VersionNode:
    version: SchemaVersion
    version_type: VersionType
    capabilities: frozenset[CapabilityId] = field(default_factory=frozenset)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))
        if not isinstance(self.capabilities, frozenset):
            object.__setattr__(self, "capabilities", frozenset(self.capabilities))
        self._validate()

    def _validate(self) -> None:
        """Invariant 验证钩子（当前为空，预留扩展）。"""
        pass


# ============================================================================
# Upcaster Protocol
# ============================================================================

@runtime_checkable
class Upcaster(Protocol):
    def __call__(self, snapshot: RawSnapshot, context: MigrationContext) -> RawSnapshot:
        ...


# ============================================================================
# MigrationEdge
# ============================================================================

@dataclass(frozen=True)
class MigrationEdge:
    from_version: SchemaVersion
    to_version: SchemaVersion
    upcaster: Upcaster




# ============================================================================
# MigrationContext
# ============================================================================

@dataclass(frozen=True)
class MigrationContext:
    clock: Clock = field(default_factory=FixedClock)
    rng: RandomProvider = field(default_factory=FixedRandom)
    environment: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "environment", freeze_mapping(self.environment))