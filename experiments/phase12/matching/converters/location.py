from src.writing.planning_contract import StateChange
from ..state_expectation import ExpectedStateChange
from ..severity import Severity
from .protocol import StateChangeConverter


class LocationConverter:
    supported_type = "location"

    def convert(self, state_change: StateChange) -> ExpectedStateChange:
        id_ = getattr(state_change, 'id', None) or f"location_{state_change.actor}_{state_change.location}"
        return ExpectedStateChange(
            id=id_,
            field="location",
            value=state_change.location,
            target=state_change.actor,
            operation="set",
            description=f"{state_change.actor} 移动到 {state_change.location}",
            severity=Severity.MEDIUM
        )