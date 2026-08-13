# src/writing/snapshot/encoder_utils.py

from dataclasses import fields, is_dataclass
from typing import Any, Dict


def encode_dataclass(obj: Any) -> Dict[str, Any]:
    if not is_dataclass(obj):
        raise TypeError(f"Expected dataclass, got {type(obj).__name__}")
    result = {}
    for f in fields(obj):
        value = getattr(obj, f.name)
        # 由 SnapshotEncoder._default_encode 递归处理
        # 这里只做基础转换，复杂类型由上层处理
        result[f.name] = value
    return result