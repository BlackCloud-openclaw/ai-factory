# src/writing/snapshot/encoder.py

from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List
from uuid import UUID

from src.writing.snapshot.models import PipelineSnapshot
from src.writing.snapshot.encoder_registry import EncoderRegistry


class SnapshotEncoder:
    def encode(self, snapshot: PipelineSnapshot) -> Dict[str, Any]:
        result = {}
        for field in fields(snapshot):
            value = getattr(snapshot, field.name)
            encoder = EncoderRegistry.get(field.name)
            if encoder is not None:
                result[field.name] = encoder.encode(value)
            else:
                result[field.name] = self._default_encode(value)
        return result
    
    def _default_encode(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, UUID):
            return str(value)
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, Enum):
            return value.name
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [self._default_encode(v) for v in value]
        if isinstance(value, Mapping):
            return {str(k): self._default_encode(v) for k, v in value.items()}
        if is_dataclass(value):
            return self._encode_dataclass(value)
        raise TypeError(f"Unsupported type: {type(value).__name__}")
        
    def _encode_dataclass(self, obj: Any) -> Dict[str, Any]:
        result = {}
        for f in fields(obj):
            val = getattr(obj, f.name)
            result[f.name] = self._default_encode(val)
        return result# 在文件顶部添加导入
from collections.abc import Sequence, Mapping
