# src/writing/common/severity.py

from enum import Enum, auto


class Severity(Enum):
    INFO = auto()
    WARNING = auto()
    ERROR = auto()