"""
ExtractionResult：Pipeline 运行结果（不可变）
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional

from ..corpus.models import CorpusSample


@dataclass(frozen=True)
class ExtractionStats:
    """不可变的统计信息（含错误列表）"""
    total_records: int = 0
    normalized: int = 0
    classified: int = 0
    built: int = 0
    validated: int = 0
    exported: int = 0
    skipped: int = 0
    invalid: int = 0
    errors: Tuple[str, ...] = field(default_factory=tuple)


class MutableExtractionStats:
    """Pipeline 内部使用的可变统计累加器"""
    def __init__(self):
        self.total_records = 0
        self.normalized = 0
        self.classified = 0
        self.built = 0
        self.validated = 0
        self.exported = 0
        self.skipped = 0
        self.invalid = 0
        self._errors: List[str] = []

    def add_error(self, error: str) -> None:
        self._errors.append(error)

    def freeze(self) -> ExtractionStats:
        """唯一转换出口：返回不可变统计对象"""
        return ExtractionStats(
            total_records=self.total_records,
            normalized=self.normalized,
            classified=self.classified,
            built=self.built,
            validated=self.validated,
            exported=self.exported,
            skipped=self.skipped,
            invalid=self.invalid,
            errors=tuple(self._errors),
        )


@dataclass(frozen=True)
class ExtractionResult:
    """不可变的 Pipeline 运行结果"""
    stats: ExtractionStats
    samples: Tuple[CorpusSample, ...] = field(default_factory=tuple)

    @property
    def success(self) -> bool:
        return not self.stats.errors and self.stats.validated > 0

    @property
    def total_records(self) -> int:
        return self.stats.total_records

    @property
    def exported(self) -> int:
        return self.stats.exported

    def __str__(self) -> str:
        s = self.stats
        lines = [
            "ExtractionResult:",
            f"  total_records: {s.total_records}",
            f"  normalized: {s.normalized}",
            f"  classified: {s.classified}",
            f"  built: {s.built}",
            f"  validated: {s.validated}",
            f"  exported: {s.exported}",
            f"  skipped: {s.skipped}",
            f"  invalid: {s.invalid}",
            f"  samples: {len(self.samples)}",
        ]
        if s.errors:
            lines.append(f"  errors: {len(s.errors)}")
        return "\n".join(lines)