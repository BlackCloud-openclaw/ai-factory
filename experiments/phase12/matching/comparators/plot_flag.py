from ..state_expectation import ExpectedStateChange
from ..state_match_result import StateMatchResult
from ..snapshot_accessor import SnapshotAccessor
from ..severity import Severity
from .protocol import StateFieldComparator


class PlotFlagComparator(StateFieldComparator):
    field = "plot_flag"

    def compare(
        self,
        expected: ExpectedStateChange,
        before: SnapshotAccessor,
        after: SnapshotAccessor,
    ) -> StateMatchResult:
        flag = expected.target or expected.field
        expected_value = expected.value

        if not after.exists:
            return StateMatchResult(
                field=self.field,
                expectation_id=expected.id,
                matched=False,
                expected=expected_value,
                actual=None,
                strategy="missing_snapshot",
                confidence=0.0,
                severity=Severity.CRITICAL,
                details={"message": "After snapshot missing"},
            )

        actual_flag = after.get_plot_flag(flag)
        matched = (actual_flag == expected_value)

        return StateMatchResult(
            field=self.field,
            expectation_id=expected.id,
            matched=matched,
            expected=expected_value,
            actual=actual_flag,
            strategy="flag_change",
            confidence=1.0 if matched else 0.0,
            severity=Severity.MEDIUM,
            details={"flag": flag},
        )