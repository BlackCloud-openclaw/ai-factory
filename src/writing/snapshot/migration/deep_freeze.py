from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


def deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({k: deep_freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(deep_freeze(v) for v in value)
    if isinstance(value, set):
        return frozenset(deep_freeze(v) for v in value)
    if isinstance(value, tuple):
        return tuple(deep_freeze(v) for v in value)
    return value