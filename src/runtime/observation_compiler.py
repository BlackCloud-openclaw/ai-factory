"""
ObservationCompiler - 将 Draft 编译为 ObservationIR
Phase 7A-1: 接收 RuntimeSnapshot
"""

import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from src.runtime.snapshot import RuntimeSnapshot
from src.surfaces.definition import PatternDefinition
from src.runtime.observation_ir import ObservationIR, SentenceSpan, PatternSpan, DocumentMetadata


class ObservationCompiler:
    """
    ObservationCompiler 是 ObservationIR 的唯一生产者。
    
    Phase 7A-1 修改：
    - 接收 RuntimeSnapshot 而非单独的 Surface
    - 从 Snapshot 中获取所有 Surface 的 Pattern
    - 不依赖 Registry
    """
    
    COMPILER_VERSION = "1.0.0"

    def __init__(self):
        self._matcher_registry = {
            "keyword": self._match_keyword,
            "regex": self._match_regex,
            "quotation": self._match_quotation,
            "builtin.quotation": self._match_quotation,  # 添加别名
        }

    def compile(self, draft: str, snapshot: RuntimeSnapshot) -> ObservationIR:
        """
        编译 Draft 为 ObservationIR
        
        :param draft: 原始文本
        :param snapshot: RuntimeSnapshot（包含所有 Surface 和配置）
        """
        # 1. 分句
        sentences = self._split_sentences(draft)
        
        # 2. 提取所有 Pattern（遍历所有 Surface）
        patterns: List[PatternSpan] = []
        for surface in snapshot.surfaces:
            for pattern_def in surface.observation.patterns:
                matches = self._extract_pattern(draft, pattern_def, sentences)
                patterns.extend(matches)
        
        # 3. 按位置排序
        patterns.sort(key=lambda p: (p.start, p.end))
        for idx, p in enumerate(patterns):
            p.id = f"P_{idx}"
        
        # 4. 构建 IR
        metadata = DocumentMetadata(
            total_chars=len(draft),
            sentence_count=len(sentences),
            pattern_count=len(patterns),
            created_at=datetime.utcnow().isoformat() + "Z",
        )
        
        return ObservationIR(
            version=1,
            compiler_version=self.COMPILER_VERSION,
            source_hash=hashlib.sha256(draft.encode("utf-8")).hexdigest(),
            metadata=metadata,
            sentences=sentences,
            patterns=patterns,
        )
    
    def _extract_pattern(self, draft: str, pattern_def: PatternDefinition, sentences: List[SentenceSpan]) -> List[PatternSpan]:
        """根据 PatternDefinition 提取匹配"""
        print(f"DEBUG: pattern_def.matcher = '{pattern_def.matcher}'")
        print(f"DEBUG: available matchers = {list(self._matcher_registry.keys())}")
        
        matcher = self._matcher_registry.get(pattern_def.matcher)
        if not matcher:
            print(f"DEBUG: No matcher found for '{pattern_def.matcher}'")
            return []
        print(f"DEBUG: Matcher found for '{pattern_def.matcher}'")
        return matcher(draft, pattern_def, sentences)
    
    def _match_keyword(self, draft: str, pattern_def: PatternDefinition, sentences: List[SentenceSpan]) -> List[PatternSpan]:
        """关键词匹配"""
        results: List[PatternSpan] = []
        keywords = pattern_def.config.get("keywords", [])
        if not keywords:
            return results
        
        for sent in sentences:
            for keyword in keywords:
                start_idx = 0
                while True:
                    pos = sent.text.find(keyword, start_idx)
                    if pos == -1:
                        break
                    abs_start = sent.start + pos
                    abs_end = abs_start + len(keyword)
                    results.append(
                        PatternSpan(
                            id=f"P_temp",
                            start=abs_start,
                            end=abs_end,
                            text=keyword,
                            pattern_type=pattern_def.name,
                            sentence_id=sent.id,
                        )
                    )
                    start_idx = pos + 1
        return results
    
    def _match_regex(self, draft: str, pattern_def: PatternDefinition, sentences: List[SentenceSpan]) -> List[PatternSpan]:
        """正则匹配（暂未实现）"""
        return []
        
    def _match_quotation(self, draft: str, pattern_def: PatternDefinition, sentences: List[SentenceSpan]) -> List[PatternSpan]:
        """
        引号匹配：使用正则匹配所有引号内容
        """
        import re
        results = []
        # 从 config 获取 patterns，如果没有则使用默认
        patterns = pattern_def.config.get("patterns", [r'「.*?」', r'『.*?』', r'“.*?”', r'".*?"'])
        for pattern in patterns:
            for match in re.finditer(pattern, draft):
                full_text = match.group(0)
                abs_start = match.start()
                abs_end = match.end()
                # 找到所属句子
                sent_id = None
                for sent in sentences:
                    if sent.start <= abs_start < sent.end:
                        sent_id = sent.id
                        break
                if sent_id is None:
                    continue
                results.append(
                    PatternSpan(
                        id=f"P_temp",
                        start=abs_start,
                        end=abs_end,
                        text=full_text,
                        pattern_type=pattern_def.name,
                        sentence_id=sent_id,
                    )
                )
        return results
        
    def _split_sentences(self, text: str) -> List[SentenceSpan]:
        """分句逻辑（保持与 Phase 6 一致）"""
        if not text:
            return []
        delimiters = {'。', '！', '？', '\n'}
        sentences: List[SentenceSpan] = []
        start = 0
        counter = 0
        for i, ch in enumerate(text):
            if ch in delimiters:
                end = i + 1
                sent_text = text[start:end]
                if sent_text:
                    sentences.append(
                        SentenceSpan(
                            id=f"S_{counter}",
                            start=start,
                            end=end,
                            text=sent_text,
                        )
                    )
                    counter += 1
                start = end
        if start < len(text):
            sent_text = text[start:]
            if sent_text:
                sentences.append(
                    SentenceSpan(
                        id=f"S_{counter}",
                        start=start,
                        end=len(text),
                        text=sent_text,
                    )
                )
        return sentences