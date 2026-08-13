from src.writing.planning_contract import StateChange
from ..state_expectation import ExpectedStateChange
from ..severity import Severity
from .protocol import StateChangeConverter


class HpConverter:
    supported_type = "hp"

    def convert(self, state_change: StateChange) -> ExpectedStateChange:
        id_ = getattr(state_change, 'id', None) or f"hp_{state_change.actor}_{state_change.new_hp}"
        return ExpectedStateChange(
            id=id_,
            field="hp",
            value=state_change.new_hp,
            target=state_change.actor,
            operation="set",
            description=f"{state_change.actor} HP 变为 {state_change.new_hp}",
            severity=Severity.MEDIUM
        )