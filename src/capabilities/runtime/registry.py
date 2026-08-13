# src/capabilities/runtime/registry.py
"""
Phase 11.2.1: RuntimeCapabilityRegistry
"""

from typing import Dict
from types import MappingProxyType

from ..spec import CapabilitySpec
from .protocol import RuntimeCapability
from .frozen import FrozenRuntimeCapabilityRegistry


class RuntimeCapabilityRegistry:
    def __init__(self):
        self._registry: Dict[str, RuntimeCapability] = {}
        self._frozen: bool = False

    def register(self, spec: CapabilitySpec, capability: RuntimeCapability) -> None:
        if self._frozen:
            raise RuntimeError("RuntimeCapabilityRegistry is frozen")

        if spec.id in self._registry:
            raise ValueError(
                f"Runtime capability '{spec.id}' already registered"
            )

        self._registry[spec.id] = capability

    def require(self, capability_id: str) -> RuntimeCapability:
        try:
            return self._registry[capability_id]
        except KeyError:
            raise KeyError(
                f"Runtime capability not found: {capability_id}"
            )

    def has(self, capability_id: str) -> bool:
        return capability_id in self._registry

    def list_ids(self) -> tuple[str, ...]:
        return tuple(self._registry.keys())

    def freeze(self) -> FrozenRuntimeCapabilityRegistry:
        if not self._registry:
            raise RuntimeError("Cannot freeze empty RuntimeCapabilityRegistry")

        self._frozen = True
        return FrozenRuntimeCapabilityRegistry(
            MappingProxyType(self._registry)
        )