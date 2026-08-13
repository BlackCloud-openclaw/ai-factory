from typing import Dict, Optional, Sequence
from .protocol import StateFieldComparator


class StateFieldComparatorRegistry:
    def __init__(self):
        self._comparators: Dict[str, StateFieldComparator] = {}

    def register(self, comparator: StateFieldComparator) -> None:
        if comparator.field in self._comparators:
            raise ValueError(f"Comparator for '{comparator.field}' already registered")
        self._comparators[comparator.field] = comparator

    def get(self, field: str) -> Optional[StateFieldComparator]:
        return self._comparators.get(field)

    def contains(self, field: str) -> bool:
        return field in self._comparators

    def all(self) -> Sequence[StateFieldComparator]:
        return tuple(self._comparators.values())

    @classmethod
    def with_defaults(cls) -> "StateFieldComparatorRegistry":
        from .realm import RealmComparator
        from .hp import HpComparator
        from .inventory import InventoryComparator
        from .location import LocationComparator
        from .plot_flag import PlotFlagComparator
        from .relationship import RelationshipComparator
        registry = cls()
        registry.register(RealmComparator())
        registry.register(HpComparator())
        registry.register(InventoryComparator())
        registry.register(LocationComparator())
        registry.register(PlotFlagComparator())
        registry.register(RelationshipComparator())
        return registry