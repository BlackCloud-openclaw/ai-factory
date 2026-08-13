"""
Phase 12.1A-3：Gold Corpus 提取 Pipeline
"""

from .models import (
    RawFailureRecord,
    NormalizedFailure,
    FailureSource,
    ClassifiedFailure,
)
from .provider import FailureProvider
from .log_provider import LogFailureProvider
from .normalizer import FailureNormalizer
from .classifier import FailureClassifier
from .builder import CorpusSampleBuilder
from .validator import SchemaValidator
from .exporter import YamlExporter
from .repository import CorpusRepository
from .pipeline import ExtractionPipeline, SampleIdGenerator, SequentialSampleIdGenerator
from .config import ExtractionConfig
from .result import ExtractionResult, ExtractionStats, MutableExtractionStats

__all__ = [
    "RawFailureRecord",
    "NormalizedFailure",
    "ClassifiedFailure",
    "FailureSource",
    "FailureProvider",
    "LogFailureProvider",
    "FailureNormalizer",
    "FailureClassifier",
    "CorpusSampleBuilder",
    "SchemaValidator",
    "YamlExporter",
    "CorpusRepository",
    "ExtractionPipeline",
    "SampleIdGenerator",
    "SequentialSampleIdGenerator",
    "ExtractionConfig",
    "ExtractionResult",
    "ExtractionStats",
    "MutableExtractionStats",
]