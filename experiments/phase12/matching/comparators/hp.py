from ..state_expectation import ExpectedStateChange
from ..state_match_result import StateMatchResult
from ..snapshot_accessor import SnapshotAccessor
from ..severity import Severity
from .protocol import StateFieldComparator


class HpComparator(StateFieldComparator):
    field = "hp"

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

        before_hp = before.get_hp(target)
        after_hp = after.get_hp(target)

        if before_hp is None and after_hp is None:
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

        matched = (after_hp == expected_value)

        return StateMatchResult(
            field=self.field,
            expectation_id=expected.id,
            matched=matched,
            expected=expected_value,
            actual=after_hp,
            strategy="hp_change",
            confidence=1.0 if matched else 0.0,
            severity=Severity.MEDIUM,
            details={"before": before_hp, "after": after_hp},
        )