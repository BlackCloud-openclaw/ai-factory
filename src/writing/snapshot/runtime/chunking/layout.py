# src/writing/snapshot/runtime/chunking/layout.py
"""
B3.3: ChunkLayout — 分块策略布局描述（Manifest 的一部分）
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class ChunkLayout:
    """分块策略的布局描述，存储在 Manifest 中。"""

    algorithm: str  # "fixed", "cdc", "rolling-hash", ...
    target_chunk_size: int | None = None
    min_chunk_size: int | None = None
    max_chunk_size: int | None = None

    # 扩展参数（如 window_size, hash_bits, 等），由具体 Strategy 填充
    parameters: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))

    def to_mapping(self) -> dict[str, Any]:
        """序列化为 JSON 兼容的 dict。"""
        return {
            "algorithm": self.algorithm,
            "target_chunk_size": self.target_chunk_size,
            "min_chunk_size": self.min_chunk_size,
            "max_chunk_size": self.max_chunk_size,
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ChunkLayout":
        """从 JSON 反序列化。"""
        return cls(
            algorithm=data.get("algorithm", "fixed"),
            target_chunk_size=data.get("target_chunk_size"),
            min_chunk_size=data.get("min_chunk_size"),
            max_chunk_size=data.get("max_chunk_size"),
            parameters=data.get("parameters", {}),
        )