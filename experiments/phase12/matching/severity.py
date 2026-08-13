from enum import Enum


class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @classmethod
    def from_string(cls, value: str) -> "Severity":
        try:
            return cls(value.lower())
        except ValueError:
            return cls.MEDIUM