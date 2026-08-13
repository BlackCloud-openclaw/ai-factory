"""
Corpus 模块 - Phase 12.1A-2
"""

from .models import (
    Corpus,
    CorpusMetadata,
    CorpusSample,
    CorpusArtifacts,
    ExpectedResult,
)
from .failure_mode import FailureMode, Difficulty, ExpectationType
from .loader import CorpusLoader, load_samples, load_corpus
from .adapter import RuntimeArtifactAdapter, RuntimeSnapshotDeserializer, NarrativeEventDeserializer, PlanningContractDeserializer
from .factory import ContextFactory
from .judge_context_factory import JudgeContextFactory

__all__ = [
    "Corpus",
    "CorpusMetadata",
    "CorpusSample",
    "CorpusArtifacts",
    "ExpectedResult",
    "FailureMode",
    "Difficulty",
    "ExpectationType",
    "CorpusLoader",
    "load_samples",
    "load_corpus",
    "RuntimeArtifactAdapter",
    "RuntimeSnapshotDeserializer",
    "NarrativeEventDeserializer",
    "PlanningContractDeserializer",
    "ContextFactory",
    "JudgeContextFactory",
]