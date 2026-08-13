# src/writing/validation/matchers.py
"""
Matcher Pipeline - Exact, Normalized, Keyword Coverage (使用 n-gram)
Phase 13.2.3B Patch-1.3 Final
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Set


@dataclass
class MatcherResult:
    matched: bool
    confidence: float = 0.0
    matcher: str = ""
    matched_text: str = ""
    evidence_id: str = ""


class Matcher(ABC):
    @abstractmethod
    def match(self, event: str, text: str) -> MatcherResult:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class ExactMatcher(Matcher):
    @property
    def name(self) -> str:
        return "exact"

    def match(self, event: str, text: str) -> MatcherResult:
        if event in text:
            return MatcherResult(
                matched=True,
                confidence=1.0,
                matcher=self.name,
                matched_text=event,
            )
        return MatcherResult(matched=False, matcher=self.name)


class NormalizedMatcher(Matcher):
    """去除停用词后检查事件是否作为子串出现（保留原始逻辑）"""
    STOP_WORDS = {
        "的", "了", "是", "在", "和", "与", "或", "但", "而", "被", "把", "让",
        "会", "能", "可以", "将", "就", "也", "还", "都", "不", "没", "有",
        "这", "那", "一", "我", "你", "他", "她", "它", "们", "地", "得",
        "着", "过", "对", "为", "以", "到", "去", "说", "看", "听", "想",
        "知道", "觉得", "看见", "听见"
    }

    @property
    def name(self) -> str:
        return "normalized"

    def match(self, event: str, text: str) -> MatcherResult:
        event_clean = self._remove_stopwords(event)
        text_clean = self._remove_stopwords(text)

        if not event_clean:
            if event in text:
                return MatcherResult(
                    matched=True,
                    confidence=0.8,
                    matcher=self.name,
                    matched_text=event,
                )
            return MatcherResult(matched=False, matcher=self.name)

        if event_clean in text_clean:
            return MatcherResult(
                matched=True,
                confidence=0.9,
                matcher=self.name,
                matched_text=event_clean,
            )
        return MatcherResult(matched=False, matcher=self.name)

    def _remove_stopwords(self, text: str) -> str:
        for word in self.STOP_WORDS:
            text = text.replace(word, '')
        return text


class KeywordCoverageMatcher(Matcher):
    """
    基于字符 n-gram 的关键词覆盖率。
    使用事件文本的所有 3-gram 作为关键词，计算在正文中的覆盖率。
    阈值默认 0.6。
    """
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "keyword_coverage"

    def match(self, event: str, text: str) -> MatcherResult:
        # 提取事件的所有 3-gram
        event_ngrams = self._extract_ngrams(event, min_len=3, max_len=3)
        if not event_ngrams:
            # 事件太短（少于3字符），回退到精确匹配
            if event in text:
                return MatcherResult(
                    matched=True,
                    confidence=1.0,
                    matcher=self.name,
                    matched_text=event,
                )
            return MatcherResult(matched=False, matcher=self.name)

        # 提取正文的所有 2-gram 和 3-gram（为了覆盖更短的关键词）
        text_ngrams = self._extract_ngrams(text, min_len=2, max_len=3)
        if not text_ngrams:
            return MatcherResult(matched=False, matcher=self.name)

        # 只使用 3-gram 计算覆盖率（稳定且精确）
        event_set = set(event_ngrams)
        text_set = set(text_ngrams)  # 包含 2-gram 和 3-gram，但匹配时只比较 3-gram 存在性
        # 实际上，我们需要检查事件 3-gram 是否在文本 3-gram 集合中
        # 但文本 3-gram 可能缺失，所以改用文本所有 n-gram 集合
        matched = sum(1 for ng in event_set if ng in text_set)
        coverage = matched / len(event_set)

        if coverage >= self.threshold:
            return MatcherResult(
                matched=True,
                confidence=coverage,
                matcher=self.name,
                matched_text=" ".join(sorted(event_set)[:5]),  # 稳定排序
            )
        return MatcherResult(matched=False, confidence=coverage, matcher=self.name)

    def _extract_ngrams(self, text: str, min_len: int = 2, max_len: int = 3) -> List[str]:
        """
        提取所有长度在 [min_len, max_len] 范围内的连续中文字符子串。
        返回排序列表以保证确定性。
        """
        ngrams = set()
        chars = list(text)
        for i in range(len(chars)):
            for l in range(min_len, max_len + 1):
                if i + l <= len(chars):
                    ngram = ''.join(chars[i:i+l])
                    # 仅保留纯中文字符串
                    if re.match(r'^[\u4e00-\u9fff]+$', ngram):
                        ngrams.add(ngram)
        return sorted(ngrams)