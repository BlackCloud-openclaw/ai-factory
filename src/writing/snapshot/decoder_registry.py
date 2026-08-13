# src/writing/snapshot/decoder_registry.py

from typing import Dict, Optional, Any, Protocol, runtime_checkable


@runtime_checkable
class Decoder(Protocol):
    field_name: str

    def decode(self, data: Dict[str, Any]) -> Any:
        ...


class DecoderRegistry:
    _decoders: Dict[str, Decoder] = {}

    @classmethod
    def register(cls, field_name: str, decoder: Decoder) -> None:
        cls._decoders[field_name] = decoder

    @classmethod
    def get(cls, field_name: str) -> Optional[Decoder]:
        return cls._decoders.get(field_name)