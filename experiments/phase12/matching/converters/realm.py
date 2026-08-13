from src.writing.planning_contract import StateChange
from ..state_expectation import ExpectedStateChange
from ..severity import Severity
from .protocol import StateChangeConverter


class RealmConverter:
    supported_type = "realm"

    def convert(self, state_change: StateChange) -> ExpectedStateChange:
        # 生成稳定的 ID
        id_ = getattr(state_change, 'id', None) or f"realm_{state_change.actor}_{state_change.to_major_realm}"
        return ExpectedStateChange(
            id=id_,
            field="realm",
            value=state_change.to_major_realm,
            target=state_change.actor,
            operation="set",
            description=f"{state_change.actor} 突破到 {state_change.to_major_realm}{state_change.to_minor_stage}层",
            severity=Severity.HIGH
        )