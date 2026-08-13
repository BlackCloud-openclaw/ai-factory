"""
RuntimeArtifactAdapter：通过注入 Deserializer 解耦 Runtime
"""

from typing import Optional, List, Any, Mapping, Protocol

from src.writing.planning_contract import PlanningContract
from src.writing.events import NarrativeEvent
from src.writing.snapshot.runtime.models import RuntimeSnapshot


class PlanningContractDeserializer(Protocol):
    def deserialize(self, data: Mapping[str, Any]) -> PlanningContract:
        ...


class RuntimeSnapshotDeserializer(Protocol):
    def deserialize(self, data: Mapping[str, Any]) -> RuntimeSnapshot:
        ...


class NarrativeEventDeserializer(Protocol):
    def deserialize(self, data: Mapping[str, Any]) -> NarrativeEvent:
        ...


class RuntimeArtifactAdapter:
    """通过注入 Deserializer 实现 Runtime 转换，与 Phase12 Factory 解耦"""

    def __init__(
        self,
        contract_deserializer: Optional[PlanningContractDeserializer] = None,
        snapshot_deserializer: Optional[RuntimeSnapshotDeserializer] = None,
        event_deserializer: Optional[NarrativeEventDeserializer] = None,
    ):
        self._contract_deserializer = contract_deserializer
        self._snapshot_deserializer = snapshot_deserializer
        self._event_deserializer = event_deserializer

    def to_planning_contract(self, data: Optional[Mapping[str, Any]]) -> Optional[PlanningContract]:
        if data is None or self._contract_deserializer is None:
            return None
        return self._contract_deserializer.deserialize(data)

    def to_snapshot(self, data: Optional[Mapping[str, Any]]) -> Optional[RuntimeSnapshot]:
        if data is None or self._snapshot_deserializer is None:
            return None
        return self._snapshot_deserializer.deserialize(data)

    def to_events(self, data: Optional[List[Mapping[str, Any]]]) -> List[NarrativeEvent]:
        if data is None or self._event_deserializer is None:
            return []
        return [self._event_deserializer.deserialize(item) for item in data]