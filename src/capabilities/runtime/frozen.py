# src/capabilities/runtime/frozen.py

from types import MappingProxyType

from .protocol import RuntimeCapability


class FrozenRuntimeCapabilityRegistry:
    def __init__(self, registry: MappingProxyType[str, RuntimeCapability]):
        self._registry = registry

    def require(self, capability_id: str) -> RuntimeCapability:
        if capability_id not in self._registry:
            raise KeyError(
                f"Runtime capability not found: {capability_id}"
            )
        return self._registry[capability_id]

    def has(self, capability_id: str) -> bool:
        return capability_id in self._registry

    def list_ids(self) -> tuple[str, ...]:
        return tuple(self._registry.keys())

    @property
    def is_frozen(self) -> bool:
        return True