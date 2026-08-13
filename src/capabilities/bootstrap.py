# src/capabilities/bootstrap.py

from src.capabilities import CapabilityRegistry
from src.capabilities.builtin import (
    KEYWORD_SPEC,
    KEYWORD_IMPL,
    QUOTATION_SPEC,
    QUOTATION_IMPL,
)


def create_default_registry() -> CapabilityRegistry:
    """创建包含内置 Capability 的默认 Registry。"""
    return CapabilityRegistry(
        specs={
            "builtin.keyword": KEYWORD_SPEC,
            "builtin.quotation": QUOTATION_SPEC,
        },
        implementations={
            "builtin.keyword": KEYWORD_IMPL,
            "builtin.quotation": QUOTATION_IMPL,
        },
    )