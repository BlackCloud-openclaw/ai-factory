# src/narrative/_utils.py

from uuid import UUID, uuid4
from typing import Any


def parse_uuid(value: Any, field_name: str = "id") -> UUID:
    if value is None:
        return uuid4()
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid UUID for {field_name}: {value}") from e