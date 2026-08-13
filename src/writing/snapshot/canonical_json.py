# src/writing/snapshot/canonical_json.py

import json
from typing import Any, Mapping


def dumps(data: Mapping[str, Any]) -> bytes:
    """
    输出 Canonical JSON，满足：
    - UTF-8
    - sort_keys=True
    - separators=(",", ":")
    - indent=2
    - ensure_ascii=False
    """
    return (
        json.dumps(
            data,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            indent=2,
        )
        .encode("utf-8")
    )


def loads(data: bytes) -> dict:
    return json.loads(data.decode("utf-8"))