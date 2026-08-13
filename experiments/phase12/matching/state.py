from typing import Protocol, Sequence, Optional
import dataclasses
from .state_expectation import StateExpectation
from .state_match_result import StateMatchResult
from .snapshot_accessor import RuntimeSnapshotAccessor
from .comparators import StateFieldComparatorRegistry
from .severity import Severity
from src.writing.snapshot.runtime.models import RuntimeSnapshot


class StateMatcher(Protocol):
    def compare(
        self,
        expected: StateExpectation,
        before: Optional[RuntimeSnapshot],
        after: Optional[RuntimeSnapshot],
    ) -> Sequence[StateMatchResult]:
        ...


class RuleStateMatcher:
    def __init__(self, comparator_registry: Optional[StateFieldComparatorRegistry] = None):
        self._registry = comparator_registry or StateFieldComparatorRegistry.with_defaults()

    def compare(
        self,
        expected: StateExpectation,
        before: Optional[RuntimeSnapshot],
        after: Optional[RuntimeSnapshot],
    ) -> Sequence[StateMatchResult]:
        before_acc = RuntimeSnapshotAccessor(before)
        after_acc = RuntimeSnapshotAccessor(after)
        results = []
        for change in expected.changes:
            comp = self._registry.get(change.field)
            if comp is None:
                results.append(StateMatchResult(
                    field=change.field,
                    expectation_id=change.id,
                    matched=False,
                    expected=change.value,
                    actual=None,
                    strategy="unsupported_field",
                    confidence=0.0,
                    severity=Severity.CRITICAL,
                    details={"message": f"No comparator for field '{change.field}'"},
                ))
            else:
                result = comp.compare(change, before_acc, after_acc)
                results.append(dataclasses.replace(
                    result,
                    expectation_id=change.id,
                    confidence=1.0 if result.matched else 0.0,
                    severity=change.severity,
                ))
        return results