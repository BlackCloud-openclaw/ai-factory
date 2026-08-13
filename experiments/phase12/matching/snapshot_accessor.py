from typing import Protocol, Optional, List, Any, runtime_checkable
from src.writing.snapshot.runtime.models import RuntimeSnapshot


@runtime_checkable
class SnapshotAccessor(Protocol):
    @property
    def exists(self) -> bool: ...

    def get_character_field(self, actor: str, field: str) -> Optional[Any]:
        ...

    def get_global_field(self, name: str) -> Optional[Any]:
        ...

    def get_character_realm(self, actor: str) -> Optional[str]:
        ...

    def get_hp(self, actor: str) -> Optional[int]:
        ...

    def get_mp(self, actor: str) -> Optional[int]:
        ...

    def get_inventory(self, actor: str) -> List[str]:
        ...

    def get_location(self, actor: str) -> Optional[str]:
        ...

    def get_plot_flag(self, flag: str) -> Optional[Any]:
        ...

    def get_relationship(self, key: str) -> Optional[int]:
        ...


class RuntimeSnapshotAccessor:
    def __init__(self, snapshot: Optional[RuntimeSnapshot]):
        self._snapshot = snapshot

    @property
    def exists(self) -> bool:
        return self._snapshot is not None

    def get_character_field(self, actor: str, field: str) -> Optional[Any]:
        if not self._snapshot:
            return None
        return self._snapshot.characters.get(actor, {}).get(field)

    def get_global_field(self, name: str) -> Optional[Any]:
        if not self._snapshot:
            return None
        return self._snapshot.global_flags.get(name)

    def get_character_realm(self, actor: str) -> Optional[str]:
        return self.get_character_field(actor, "realm")

    def get_hp(self, actor: str) -> Optional[int]:
        return self.get_character_field(actor, "hp")

    def get_mp(self, actor: str) -> Optional[int]:
        return self.get_character_field(actor, "mp")

    def get_inventory(self, actor: str) -> List[str]:
        if not self._snapshot:
            return []
        return self._snapshot.characters.get(actor, {}).get("inventory", [])

    def get_location(self, actor: str) -> Optional[str]:
        return self.get_character_field(actor, "location")

    def get_plot_flag(self, flag: str) -> Optional[Any]:
        return self.get_global_field(flag)

    def get_relationship(self, key: str) -> Optional[int]:
        if not self._snapshot:
            return None
        return self._snapshot.relationships.get(key)