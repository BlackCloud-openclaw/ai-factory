"""
FailureClassifier：从 NormalizedFailure 推断 Failure Mode
"""

from typing import Optional, Dict, List, Any
import yaml
from pathlib import Path

from ..corpus.failure_mode import FailureMode
from .models import NormalizedFailure, ClassifiedFailure


class FailureClassifier:
    """从 NormalizedFailure 推断 Failure Mode（配置化）"""

    def __init__(self, config: Optional[Dict[str, List[str]]] = None):
        self._config = config or self._default_config()

    def classify(self, normalized: NormalizedFailure) -> ClassifiedFailure:
        message = normalized.message.lower()

        # 关键字匹配
        for mode, keywords in self._config.items():
            for keyword in keywords:
                if keyword in message:
                    return ClassifiedFailure(
                        normalized=normalized,
                        failure_mode=FailureMode.from_string(mode),
                    )

        # 上下文补充推断
        if normalized.planning_contract:
            return ClassifiedFailure(
                normalized=normalized,
                failure_mode=FailureMode.PLANNING_EXECUTION,
            )
        elif normalized.runtime_metrics:
            return ClassifiedFailure(
                normalized=normalized,
                failure_mode=FailureMode.RUNTIME_STATE,
            )

        # 无法识别 → UNKNOWN
        return ClassifiedFailure(
            normalized=normalized,
            failure_mode=FailureMode.UNKNOWN,
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "FailureClassifier":
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            return cls(config)
        return cls()

    def _default_config(self) -> Dict[str, List[str]]:
        return {
            "scene_transition": ["continuity", "transition", "time", "location"],
            "character_state": ["character", "realm", "hp", "personality"],
            "dialogue_quality": ["dialogue", "speak", "conversation"],
            "planning_execution": ["planning", "contract", "unit", "coverage"],
            "runtime_state": ["runtime", "snapshot", "event", "retry"],
        }