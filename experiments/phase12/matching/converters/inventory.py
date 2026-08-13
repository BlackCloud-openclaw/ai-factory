from src.writing.planning_contract import StateChange
from ..state_expectation import ExpectedStateChange
from ..severity import Severity
from .protocol import StateChangeConverter


class InventoryConverter:
    supported_type = "inventory"

    def convert(self, state_change: StateChange) -> ExpectedStateChange:
        if state_change.operation not in ("acquire", "lose"):
            return None
        op = "add" if state_change.operation == "acquire" else "remove"
        id_ = getattr(state_change, 'id', None) or f"inventory_{state_change.actor}_{state_change.item}_{op}"
        return ExpectedStateChange(
            id=id_,
            field="inventory",
            value=state_change.item,
            target=state_change.actor,
            operation=op,
            description=f"{state_change.actor} {'获得' if state_change.operation == 'acquire' else '失去'} {state_change.item}",
            severity=Severity.MEDIUM
        )