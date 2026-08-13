from src.writing.planning_contract import StateChange
from ..state_expectation import ExpectedStateChange
from ..severity import Severity
from .protocol import StateChangeConverter


class PlotFlagConverter:
    supported_type = "plot_flag"

    def convert(self, state_change: StateChange) -> ExpectedStateChange:
        id_ = getattr(state_change, 'id', None) or f"plot_flag_{state_change.name}_{state_change.value}"
        return ExpectedStateChange(
            id=id_,
            field="plot_flag",
            value=state_change.value,
            target=state_change.name,
            operation="set",
            description=f"剧情标记 {state_change.name} = {state_change.value}",
            severity=Severity.LOW
        )