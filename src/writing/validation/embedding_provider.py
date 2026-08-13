# src/writing/validation/embedding_provider.py
"""
Embedding Provider Protocol

Phase 13.2.3B: B-12
"""

from typing import Protocol, List
import re


class EmbeddingProvider(Protocol):
    def similarity(self, a: str, b: str) -> float:
        ...

    def batch_similarity(self, a: List[str], b: List[str]) -> List[float]:
        ...


class SentenceSplitter:
    @staticmethod
    def split(text: str) -> List[str]:
        sentences = re.split(r'[。！？；\n]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10] 