# src/writing/snapshot/serializer.py

from src.writing.snapshot.models import PipelineSnapshot
from src.writing.snapshot.encoder import SnapshotEncoder
from src.writing.snapshot.decoder import SnapshotDecoder
from src.writing.snapshot.canonical_json import dumps, loads


class JsonSerializer:
    format_name = "json"
    schema_version = "1.0"
    supports_streaming = True
    supports_compression = False

    def __init__(self):
        self.encoder = SnapshotEncoder()
        self.decoder = SnapshotDecoder()

    def serialize(self, snapshot: PipelineSnapshot) -> bytes:
        data = self.encoder.encode(snapshot)
        return dumps(data)

    def deserialize(self, data: bytes) -> PipelineSnapshot:
        raw = loads(data)
        return self.decoder.decode(raw)
from typing import Protocol, runtime_checkable

@runtime_checkable
class Serializer(Protocol):
    format_name: str
    schema_version: str
    supports_streaming: bool
    supports_compression: bool
    
    def serialize(self, snapshot: 'PipelineSnapshot') -> bytes:
        ...
    
    def deserialize(self, data: bytes) -> 'PipelineSnapshot':
        ...
