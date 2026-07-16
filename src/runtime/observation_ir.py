# src/runtime/observation_ir.py
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class Span:
    id: str
    start: int
    end: int
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "start": self.start, "end": self.end, "text": self.text}


@dataclass
class SentenceSpan(Span):
    pass


@dataclass
class PatternSpan(Span):
    pattern_type: str
    sentence_id: str

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({"pattern_type": self.pattern_type, "sentence_id": self.sentence_id})
        return base


@dataclass
class DocumentMetadata:
    total_chars: int
    sentence_count: int
    pattern_count: int
    created_at: str


@dataclass
class ObservationIR:
    version: int
    compiler_version: str
    source_hash: str
    metadata: DocumentMetadata
    sentences: List[SentenceSpan]
    patterns: List[PatternSpan]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "compiler_version": self.compiler_version,
            "source_hash": self.source_hash,
            "metadata": {
                "total_chars": self.metadata.total_chars,
                "sentence_count": self.metadata.sentence_count,
                "pattern_count": self.metadata.pattern_count,
                "created_at": self.metadata.created_at,
            },
            "sentences": [s.to_dict() for s in self.sentences],
            "patterns": [p.to_dict() for p in self.patterns],
        }