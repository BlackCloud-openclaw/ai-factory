# src/writing/stage_names.py
"""
共享的阶段名称定义（Runtime / Audit 共用）
"""

from enum import Enum


class StageName(str, Enum):
    """阶段名称（str mixin 确保 JSON 友好）。"""
    PLANNING = "planning"
    OBSERVATION = "observation"
    IR = "ir"
    PROMPT = "prompt"
    DRAFT = "draft"
    COVERAGE = "coverage"

    @classmethod
    def safe_parse(cls, value: str) -> "StageName | None":
        """安全解析，返回 None 如果未知。"""
        try:
            return cls(value)
        except ValueError:
            return None