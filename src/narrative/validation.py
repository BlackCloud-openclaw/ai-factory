# src/narrative/validation.py

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar, Mapping, Tuple, Optional, Protocol, runtime_checkable
from uuid import UUID, uuid4

from src.narrative.schema import NARRATIVE_SCHEMA_VERSION
from src.narrative._utils import parse_uuid
from src.narrative.artifact import NarrativeArtifact
from src.narrative.context import NarrativeContext
from src.narrative.intent import NarrativeIntentSet
from src.narrative.constraint import NarrativeConstraint


class ValidationSeverity(StrEnum):
    PASS = "pass"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationDomain(StrEnum):
    RUNTIME = "runtime"
    NARRATIVE = "narrative"
    SYSTEM = "system"


@dataclass(frozen=True)
class ValidationItem:
    domain: ValidationDomain
    dimension: str
    passed: bool
    severity: ValidationSeverity
    message: str
    evidence: Optional[str] = None

    SCHEMA_VERSION: ClassVar[str] = NARRATIVE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain.value,
            "dimension": self.dimension,
            "passed": self.passed,
            "severity": self.severity.value,
            "message": self.message,
            "evidence": self.evidence,
            "schema_version": self.SCHEMA_VERSION,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValidationItem":
        _ = data.get("schema_version", cls.SCHEMA_VERSION)
        return cls(
            domain=ValidationDomain(data["domain"]),
            dimension=data["dimension"],
            passed=data["passed"],
            severity=ValidationSeverity(data["severity"]),
            message=data["message"],
            evidence=data.get("evidence"),
        )


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    items: Tuple[ValidationItem, ...] = field(default_factory=tuple)
    summary: Mapping[str, Any] = field(default_factory=dict)
    validation_id: UUID = field(default_factory=uuid4)

    SCHEMA_VERSION: ClassVar[str] = NARRATIVE_SCHEMA_VERSION

    @classmethod
    def success(cls, message: str = "All validations passed") -> "ValidationResult":
        item = ValidationItem(
            domain=ValidationDomain.SYSTEM,
            dimension="overall",
            passed=True,
            severity=ValidationSeverity.PASS,
            message=message,
        )
        return cls(passed=True, items=(item,))

    @classmethod
    def failure(cls, message: str) -> "ValidationResult":
        item = ValidationItem(
            domain=ValidationDomain.SYSTEM,
            dimension="overall",
            passed=False,
            severity=ValidationSeverity.ERROR,
            message=message,
        )
        return cls(passed=False, items=(item,))

    def to_dict(self) -> dict[str, Any]:
        return {
            "validation_id": str(self.validation_id),
            "passed": self.passed,
            "items": [i.to_dict() for i in self.items],
            "summary": dict(self.summary),
            "schema_version": self.SCHEMA_VERSION,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValidationResult":
        _ = data.get("schema_version", cls.SCHEMA_VERSION)
        items = tuple(ValidationItem.from_dict(i) for i in data.get("items", []))
        return cls(
            validation_id=parse_uuid(data.get("validation_id"), "validation_id"),
            passed=data["passed"],
            items=items,
            summary=data.get("summary", {}),
        )

    def __len__(self) -> int:
        return len(self.items)

    def __bool__(self) -> bool:
        return self.passed


@runtime_checkable
class NarrativeValidator(Protocol):
    def validate(
        self,
        artifact: NarrativeArtifact,
        context: NarrativeContext,
        intents: NarrativeIntentSet,
        constraint: NarrativeConstraint,
    ) -> ValidationResult:
        ...