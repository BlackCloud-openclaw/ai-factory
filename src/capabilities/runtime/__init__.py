# src/capabilities/runtime/__init__.py
"""
Phase 11.2.1: Runtime Capability Module
"""

from .protocol import RuntimeCapability
from .registry import RuntimeCapabilityRegistry
from .frozen import FrozenRuntimeCapabilityRegistry

__all__ = [
    "RuntimeCapability",
    "RuntimeCapabilityRegistry",
    "FrozenRuntimeCapabilityRegistry",
]