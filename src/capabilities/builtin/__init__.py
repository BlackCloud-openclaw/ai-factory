# src/capabilities/builtin/__init__.py

from src.capabilities.builtin.keyword import SPEC as KEYWORD_SPEC, IMPLEMENTATION as KEYWORD_IMPL
from src.capabilities.builtin.quotation import SPEC as QUOTATION_SPEC, IMPLEMENTATION as QUOTATION_IMPL

__all__ = [
    "KEYWORD_SPEC",
    "KEYWORD_IMPL",
    "QUOTATION_SPEC",
    "QUOTATION_IMPL",
]