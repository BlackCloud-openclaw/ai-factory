# src/writing/validation/embedding_matcher.py
"""
Embedding Matcher - Adapter 化 (B-12)

P1: 修复 List 导入
"""

from typing import Optional, List  # P1
from .matchers import Matcher, MatcherResult
from .embedding_provider import EmbeddingProvider, SentenceSplitter


class EmbeddingMatcher(Matcher):
    def __init__(
        self,
        provider: EmbeddingProvider,
        threshold: float = 0.30,
        enable: bool = True,
    ):
        self.provider = provider
        self.threshold = threshold
        self.enable = enable
        self._splitter = SentenceSplitter()

    @property
    def name(self) -> str:
        return "embedding"

    def match(self, event: str, text: str) -> MatcherResult:
        if not self.enable:
            return MatcherResult(matched=False, matcher=self.name)

        if not event or not text or len(text.strip()) < 50:
            return MatcherResult(matched=False, matcher=self.name)

        sentences = self._splitter.split(text)
        if not sentences:
            return MatcherResult(matched=False, matcher=self.name)

        max_similarity = 0.0
        best_sentence = ""
        for sentence in sentences[:10]:
            try:
                sim = self.provider.similarity(event, sentence)
                if sim > max_similarity:
                    max_similarity = sim
                    best_sentence = sentence
            except Exception:
                continue

        if max_similarity >= self.threshold:
            return MatcherResult(
                matched=True,
                confidence=max_similarity,
                matcher=self.name,
                matched_text=best_sentence[:100],
            )

        return MatcherResult(
            matched=False,
            confidence=max_similarity,
            matcher=self.name,
        )


class NoOpEmbeddingProvider:
    def similarity(self, a: str, b: str) -> float:
        return 0.0

    def batch_similarity(self, a: List[str], b: List[str]) -> List[float]:
        return [0.0] * len(a)