# src/writing/snapshot/validation.py

from dataclasses import dataclass, field
from typing import List

from src.writing.common.severity import Severity


@dataclass(frozen=True)
class ValidationIssue:
    severity: Severity
    code: str
    field: str
    message: str


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    
    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]
    
    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]
    
    @property
    def infos(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.INFO]