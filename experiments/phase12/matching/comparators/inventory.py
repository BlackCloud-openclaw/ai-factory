from ..state_expectation import ExpectedStateChange
from ..state_match_result import StateMatchResult
from ..snapshot_accessor import SnapshotAccessor
from ..severity import Severity
from .protocol import StateFieldComparator


class InventoryComparator(StateFieldComparator):
    field = "inventory"

    def compare(
        self,
        expected: ExpectedStateChange,
        before: SnapshotAccessor,
        after: SnapshotAccessor,
    ) -> StateMatchResult:
        target = expected.target
        item = expected.value
        operation = expected.operation

        if not before.exists or not after.exists:
            return StateMatchResult(
                field=self.field,
                expectation_id=expected.id,
                matched=False,
                expected=item,
                actual=None,
                strategy="missing_snapshot",
                confidence=0.0,
                severity=Severity.CRITICAL,
                details={"message": "Snapshot missing"},
            )

        before_inv = set(before.get_inventory(target))
        after_inv = set(after.get_inventory(target))

        if operation == "add":
            matched = item in after_inv and item not in before_inv
        elif operation == "remove":
            matched = item not in after_inv and item in before_inv
        else:
            matched = False

        return StateMatchResult(
            field=self.field,
            expectation_id=expected.id,
            matched=matched,
            expected=f"{operation} {item}",
            actual={"before": sorted(before_inv), "after": sorted(after_inv)},
            strategy="inventory_change",
            confidence=1.0 if matched else 0.0,
            severity=Severity.MEDIUM,
            details={"before": sorted(before_inv), "after": sorted(after_inv)},
        )