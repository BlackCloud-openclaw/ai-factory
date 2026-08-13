"""
SchemaValidator：仅验证 Schema 结构，不包含质量规则
"""

from typing import List, Tuple
from ..corpus.models import CorpusSample


class SchemaValidator:
    """
    严格 Schema 验证器。

    只检查：
    - 字段存在性
    - 字段类型
    - 枚举值合法性

    不检查：
    - 业务规则（如期望非空、长度限制等）
    - 质量规则

    业务规则应放置在独立的 PolicyValidator 中（预留）。
    """

    def validate(self, sample: CorpusSample) -> Tuple[bool, List[str]]:
        errors = []

        # 必填字段
        if not sample.id:
            errors.append("id is required")
        if not sample.version:
            errors.append("version is required")
        if not sample.category:
            errors.append("category is required")

        # expected: 可以为空（Schema 层面允许）
        # 业务规则要求至少有一个 expectation，由 PolicyValidator 负责

        # 枚举值验证
        try:
            _ = sample.difficulty
        except Exception:
            errors.append("difficulty must be a valid Difficulty enum")

        try:
            _ = sample.language
        except Exception:
            errors.append("language must be a valid language code")

        # 字段类型
        if sample.scene_after is None:
            errors.append("scene_after is required")

        return len(errors) == 0, errors

    def validate_batch(self, samples: List[CorpusSample]) -> Tuple[List[CorpusSample], List[Tuple[CorpusSample, str]]]:
        valid = []
        invalid = []
        for sample in samples:
            passed, errors = self.validate(sample)
            if passed:
                valid.append(sample)
            else:
                invalid.append((sample, "; ".join(errors)))
        return valid, invalid