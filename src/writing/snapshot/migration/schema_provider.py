# src/writing/snapshot/migration/schema_provider.py
from typing import Protocol
from .version import SchemaVersion

class CurrentSchemaProvider(Protocol):
    def get(self) -> SchemaVersion:
        ...

class StaticSchemaProvider:
    def __init__(self, version: SchemaVersion):
        self._version = version
    def get(self) -> SchemaVersion:
        return self._version