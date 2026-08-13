from typing import Protocol, Mapping, Any, runtime_checkable


@runtime_checkable
class SnapshotBuilder(Protocol):
    def build(self, data: Mapping[str, Any]) -> Any:
        ...


class MockSnapshotBuilder:
    def build(self, data: Mapping[str, Any]) -> Any:
        class RuntimeSnapshotLike:
            def __init__(self, chars, flags, rels):
                self.characters = chars
                self.global_flags = flags
                self.relationships = rels

        return RuntimeSnapshotLike(
            chars=data.get("characters", {}),
            flags=data.get("global_flags", {}),
            rels=data.get("relationships", {})
        )