from ..state_expectation import ExpectedStateChange
from ..state_match_result import StateMatchResult
from ..snapshot_accessor import SnapshotAccessor
from ..severity import Severity
from .protocol import StateFieldComparator


class LocationComparator(StateFieldComparator):
    field = "location"

    def compare(
        self,
        expected: ExpectedStateChange,
        before: SnapshotAccessor,
        after: SnapshotAccessor,
    ) -> StateMatchResult:
        target = expected.target
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

        actual_location = after.get_location(target)
        matched = (actual_location == expected_value)

        return StateMatchResult(
            field=self.field,
            expectation_id=expected.id,
            matched=matched,
            expected=expected_value,
            actual=actual_location,
            strategy="location_change",
            confidence=1.0 if matched else 0.0,
            severity=Severity.MEDIUM,
            details={"target": target},
        )