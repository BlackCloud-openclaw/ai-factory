from ..state_expectation import ExpectedStateChange
from ..state_match_result import StateMatchResult
from ..snapshot_accessor import SnapshotAccessor
from ..severity import Severity
from .protocol import StateFieldComparator


class RealmComparator(StateFieldComparator):
    field = "realm"

    def compare(
        self,
        expected: ExpectedStateChange,
        before: SnapshotAccessor,
        after: SnapshotAccessor,
    ) -> StateMatchResult:
        target = expected.target
        expected_value = expected.value

        if not before.exists or not after.exists:
            return StateMatchResult(
                field=self.field,
                expectation_id=expected.id,
                matched=False,
                expected=expected_value,
                actual=None,
                strategy="missing_snapshot",
                confidence=0.0,
                severity=Severity.CRITICAL,
                details={"message": "Snapshot missing"},
            )

        before_realm = before.get_character_realm(target)
        after_realm = after.get_character_realm(target)

        if before_realm is None and after_realm is None:
            return StateMatchResult(
                field=self.field,
                expectation_id=expected.id,
                matched=False,
                expected=expected_value,
                actual=None,
                strategy="actor_not_found",
                confidence=0.0,
                severity=Severity.CRITICAL,
                details={"target": target},
            )

        matched = (after_realm == expected_value and before_realm != expected_value)

        return StateMatchResult(
            field=self.field,
            expectation_id=expected.id,
            matched=matched,
            expected=expected_value,
            actual=after_realm,
            strategy="realm_change",
            confidence=1.0 if matched else 0.0,
            severity=Severity.MEDIUM,
            details={"before": before_realm, "after": after_realm},
        )