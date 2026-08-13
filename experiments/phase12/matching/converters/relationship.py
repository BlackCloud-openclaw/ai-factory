from src.writing.planning_contract import StateChange
from ..state_expectation import ExpectedStateChange
from ..severity import Severity
from .protocol import StateChangeConverter


class RelationshipConverter:
    supported_type = "relationship"

    def convert(self, state_change: StateChange) -> ExpectedStateChange:
        key = f"{state_change.from_char}|{state_change.to_char}"
        id_ = getattr(state_change, 'id', None) or f"rel_{state_change.from_char}_{state_change.to_char}_{state_change.delta}"
        return ExpectedStateChange(
            id=id_,
            field="relationship",
            value=state_change.delta,
            target=key,
            operation="increment",
            description=f"{state_change.from_char} 与 {state_change.to_char} 关系变化 {state_change.delta}",
            severity=Severity.MEDIUM
        )