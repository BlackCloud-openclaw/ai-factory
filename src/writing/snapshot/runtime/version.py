# src/writing/snapshot/runtime/version.py
"""
B3.3: 通用语义化版本（用于 Manifest、Schema 等）
"""

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class SemanticVersion:
    """通用语义化版本 (major.minor.patch)。"""

    major: int
    minor: int
    patch: int = 0

    @classmethod
    def parse(cls, value: str) -> "SemanticVersion":
        parts = value.split(".")
        if len(parts) == 2:
            return cls(major=int(parts[0]), minor=int(parts[1]), patch=0)
        elif len(parts) == 3:
            return cls(major=int(parts[0]), minor=int(parts[1]), patch=int(parts[2]))
        raise ValueError(f"Invalid version: {value}")

    @property
    def short(self) -> str:
        return f"{self.major}.{self.minor}"

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"