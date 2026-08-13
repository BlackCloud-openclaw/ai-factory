from ..state_expectation import ExpectedStateChange
from ..state_match_result import StateMatchResult
from ..snapshot_accessor import SnapshotAccessor
from ..severity import Severity
from .protocol import StateFieldComparator


class RelationshipComparator(StateFieldComparator):
    field = "relationship"

    def compare(
        self,
        expected: ExpectedStateChange,
        before: SnapshotAccessor,
        after: SnapshotAccessor,
    ) -> StateMatchResult:
        key = expected.target
        expected_delta = expected.value

        if not before.exists or not after.exists:
            return StateMatchResult(
                field=self.field,
                expectation_id=expected.id,
                matched=False,
                expected=expected_delta,
                actual=None,
                strategy="missing_snapshot",
                confidence=0.0,
                severity=Severity.CRITICAL,
                details={"message": "Snapshot missing"},
            )

        before_rel = before.get_relationship(key) or 0
        after_rel = after.get_relationship(key) or 0
        actual_delta = after_rel - before_rel

        matched = (
            (expected_delta > 0 and actual_delta >= expected_delta) or
            (expected_delta < 0 and actual_delta <= expected_delta) or
            (expected_delta == 0 and actual_delta == 0)
        )

        return StateMatchResult(
            field=self.field,
            expectation_id=expected.id,
            matched=matched,
            expected=expected_delta,
            actual=actual_delta,
            strategy="relationship_change",
            confidence=1.0 if matched else 0.0,
            severity=Severity.MEDIUM,
            details={"before": before_rel, "after": after_rel, "key": key},
        )