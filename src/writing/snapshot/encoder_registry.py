# src/writing/snapshot/decoder_registry.py

"""
Decoder Registry — 预留接口，Release B 正式启用

Release A 使用 Builder 风格 SnapshotDecoder，
Registry 为 Release B 新增 Artifact 时提供扩展能力。
"""

from typing import Dict, Optional, Any, Protocol, runtime_checkable


@runtime_checkable
class Encoder(Protocol):
    field_name: str
    
    def encode(self, value: Any) -> Any:
        ...


class EncoderRegistry:
    _encoders: Dict[str, Encoder] = {}
    
    @classmethod
    def register(cls, field_name: str, encoder: Encoder) -> None:
        cls._encoders[field_name] = encoder
    
    @classmethod
    def get(cls, field_name: str) -> Optional[Encoder]:
        return cls._encoders.get(field_name)
    
    @classmethod
    def clear(cls) -> None:
        cls._encoders.clear()