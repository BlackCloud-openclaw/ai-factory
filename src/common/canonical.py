# src/common/canonical.py
import json
import decimal
import uuid
import unicodedata
from datetime import datetime, date
from typing import Any, Dict, List, Set, Tuple

def canonical_dumps(obj: Any, **kwargs) -> str:
    """确定性 JSON 序列化，用于哈希和比较"""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_canonical_encoder,
        **kwargs
    )

def _canonical_encoder(obj: Any) -> Any:
    if isinstance(obj, decimal.Decimal):
        return str(obj.normalize())
    if isinstance(obj, float):
        # 保留 15 位有效数字，避免 0.1+0.2 漂移
        return format(obj, ".15g")
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, str):
        return unicodedata.normalize("NFC", obj)
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, (list, tuple)):
        return [_canonical_encoder(item) for item in obj]
    if isinstance(obj, dict):
        return {_canonical_encoder(k): _canonical_encoder(v) for k, v in obj.items()}
    if hasattr(obj, "model_dump"):  # Pydantic v2
        return obj.model_dump()
    raise TypeError(f"Non-serializable type: {type(obj)}")

def canonical_hash(obj: Any) -> str:
    """计算对象的确定性哈希（SHA256）"""
    import hashlib
    serialized = canonical_dumps(obj)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()