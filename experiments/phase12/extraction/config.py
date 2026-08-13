"""
提取配置
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class ExtractionConfig:
    log_paths: List[Path] = field(default_factory=lambda: [Path("logs/ai_factory.log")])
    failure_patterns: List[str] = field(default_factory=lambda: [
        r"Low compliance",
        r"Runtime validation failed",
        r"Validation failed",
    ])
    max_records: int = 100
    output_dir: Path = Path("experiments/phase12/corpus/v1.0")
    corpus_version: str = "1.0"
    require_revision: bool = False

    @classmethod
    def default(cls) -> "ExtractionConfig":
        return cls()