from typing import Dict, Optional, Sequence
import logging
from .protocol import StateChangeConverter

logger = logging.getLogger(__name__)


class StateChangeConverterRegistry:
    def __init__(self):
        self._converters: Dict[str, StateChangeConverter] = {}

    def register(self, converter: StateChangeConverter) -> None:
        if converter.supported_type in self._converters:
            raise ValueError(f"Converter for '{converter.supported_type}' already registered")
        self._converters[converter.supported_type] = converter

    def get(self, type_name: str) -> Optional[StateChangeConverter]:
        return self._converters.get(type_name)

    def contains(self, type_name: str) -> bool:
        return type_name in self._converters

    def all(self) -> Sequence[StateChangeConverter]:
        return tuple(self._converters.values())

    @classmethod
    def with_defaults(cls) -> "StateChangeConverterRegistry":
        from .realm import RealmConverter
        from .hp import HpConverter
        from .inventory import InventoryConverter
        from .location import LocationConverter
        from .plot_flag import PlotFlagConverter
        from .relationship import RelationshipConverter
        registry = cls()
        registry.register(RealmConverter())
        registry.register(HpConverter())
        registry.register(InventoryConverter())
        registry.register(LocationConverter())
        registry.register(PlotFlagConverter())
        registry.register(RelationshipConverter())
        return registry