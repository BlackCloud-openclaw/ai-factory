# src/runtime/observation_compiler.py

import hashlib
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.runtime.snapshot import RuntimeSnapshot
from src.surfaces.definition import PatternDefinition
from src.runtime.observation_ir import ObservationIR, SentenceSpan, PatternSpan, DocumentMetadata
from src.capabilities import CapabilityLookup, CapabilityRef, CapabilityExecutionError

logger = logging.getLogger(__name__)


class ObservationCompiler:
    """
    ObservationCompiler 是 ObservationIR 的唯一生产者。

    Phase 7A-1 修改：
    - 接收 RuntimeSnapshot
    - 从 Snapshot 中获取所有 Surface 的 Pattern
    - 从 Snapshot.capability_registry 获取 CapabilityImplementation

    Phase 8 修改：
    - 删除 _matcher_registry
    - 完全通过 CapabilityLookup 工作
    - 兼容层已移至 Loader
    """

    COMPILER_VERSION = "1.0.0"

    def compile(self, draft: str, snapshot: RuntimeSnapshot) -> ObservationIR:
        """编译 Draft 为 ObservationIR"""
        sentences = self._split_sentences(draft)
        patterns: List[PatternSpan] = []
        lookup = snapshot.capability_registry

        for surface in snapshot.surfaces:
            for pattern_def in surface.observation.patterns:
                matches = self._extract_pattern(draft, pattern_def, sentences, lookup)
                patterns.extend(matches)

        # 按位置排序并分配 ID
        patterns.sort(key=lambda p: (p.start, p.end))
        for idx, p in enumerate(patterns):
            p.id = f"P_{idx}"

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

    def _extract_pattern(
        self,
        draft: str,
        pattern_def: PatternDefinition,
        sentences: List[SentenceSpan],
        lookup: CapabilityLookup,
    ) -> List[PatternSpan]:
        """
        使用 CapabilityLookup 提取 Pattern

        注意：Loader 保证 pattern_def.capability_ref 非 None。
        """
        ref = pattern_def.capability_ref

        if ref is None:
            logger.warning(
                "Pattern %s has no capability_ref, skipping (should be handled by Loader)",
                pattern_def.name,
            )
            return []

        try:
            impl = lookup.get_impl(ref)
        except Exception as e:
            # 如果 lookup 失败（CapabilityNotFoundError / CapabilityVersionError），
            # 这属于配置错误，跳过该 Pattern
            logger.warning(
                "Capability %s not available for pattern %s: %s",
                ref,
                pattern_def.name,
                e,
            )
            return []

        try:
            matches = impl.match(draft, pattern_def.config)
        except CapabilityExecutionError as e:
            # Capability 执行失败（插件边界），记录警告并跳过
            logger.warning(
                "Capability %s execution failed for pattern %s: %s",
                ref,
                pattern_def.name,
                e,
            )
            return []
        except Exception as e:
            # 其他异常（TypeError, AttributeError 等）是 Runtime bug，继续抛出
            logger.error(
                "Unexpected error in Capability %s for pattern %s: %s",
                ref,
                pattern_def.name,
                e,
                exc_info=True,
            )
            raise

        results = []
        for match in matches:
            start = match.get('start', 0)
            end = match.get('end', 0)

            # 找到所属句子
            sent_id = None
            for sent in sentences:
                if sent.start <= start < sent.end:
                    sent_id = sent.id
                    break

            if sent_id is None:
                continue

            # 兼容 pattern_type 和 type 两种字段名
            pattern_type = match.get('pattern_type') or match.get('type')
            if pattern_type is None:
                pattern_type = pattern_def.name

            results.append(
                PatternSpan(
                    id=f"P_temp",
                    start=start,
                    end=end,
                    text=match.get('text', draft[start:end]),
                    pattern_type=pattern_type,
                    sentence_id=sent_id,
                )
            )

        return results

    def _split_sentences(self, text: str) -> List[SentenceSpan]:
        """分句逻辑"""
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