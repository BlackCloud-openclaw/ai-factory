"""
Corpus 失败模式枚举
"""

from enum import Enum


class FailureMode(Enum):
    """失败模式分类，用于标记样本类型"""
    SCENE_TRANSITION = "scene_transition"
    CHARACTER_STATE = "character_state"
    CHARACTER_CONSISTENCY = "character_consistency"
    DIALOGUE_QUALITY = "dialogue_quality"
    PLANNING_EXECUTION = "planning_execution"
    RUNTIME_STATE = "runtime_state"
    READER_FLOW = "reader_flow"
    REVISION_EFFECTIVENESS = "revision_effectiveness"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, value: str) -> "FailureMode":
        for mode in cls:
            if mode.value == value:
                return mode
        raise ValueError(f"Unknown FailureMode: {value}")


class Difficulty(Enum):
    """样本难度"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

    @classmethod
    def from_string(cls, value: str) -> "Difficulty":
        for d in cls:
            if d.value == value:
                return d
        raise ValueError(f"Unknown Difficulty: {value}")


class ExpectationType(Enum):
    """期望结果类型"""
    EXACT = "exact"
    RANGE = "range"
    BOOLEAN = "boolean"
    CUSTOM = "custom"

    @classmethod
    def from_string(cls, value: str) -> "ExpectationType":
        for t in cls:
            if t.value == value:
                return t
        raise ValueError(f"Unknown ExpectationType: {value}")