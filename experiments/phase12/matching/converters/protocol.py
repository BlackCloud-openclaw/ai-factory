from typing import Protocol, Optional
from src.writing.planning_contract import StateChange
from ..state_expectation import ExpectedStateChange


class StateChangeConverter(Protocol):
    @property
    def supported_type(self) -> str: ...
    def convert(self, state_change: StateChange) -> Optional[ExpectedStateChange]: ...