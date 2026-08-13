"""
FailureNormalizer：将原始记录标准化（纯函数）
"""

from typing import Optional
from datetime import datetime

from .models import RawFailureRecord, NormalizedFailure, FailureSource


class FailureNormalizer:
    """将 RawFailureRecord 标准化为 NormalizedFailure（纯函数）"""

    def normalize(self, record: RawFailureRecord) -> Optional[NormalizedFailure]:
        """标准化单条记录"""
        if record.source == FailureSource.LOG:
            return self._normalize_log(record)
        # 其他来源可在此扩展
        return None

    def _normalize_log(self, record: RawFailureRecord) -> Optional[NormalizedFailure]:
        payload = record.payload

        # JSON 日志
        if "message" in payload:
            return NormalizedFailure(
                id=record.id or f"norm_{int(record.timestamp.timestamp())}",
                timestamp=record.timestamp,
                failure_type=payload.get("failure_type", payload.get("level", "unknown")),
                severity=payload.get("level", "error").lower(),
                message=payload.get("message", ""),
                scene_text=payload.get("scene_text"),
                planning_contract=payload.get("planning_contract"),
                events=tuple(payload.get("events", [])),  # 转为不可变 tuple
                snapshot_before=payload.get("snapshot_before"),
                snapshot_after=payload.get("snapshot_after"),
                runtime_metrics=payload.get("runtime_metrics"),
                draft_before=payload.get("draft_before"),
                draft_after=payload.get("draft_after"),
                chapter=payload.get("chapter"),
                scene_idx=payload.get("scene_idx"),
                source=FailureSource.LOG,
                tags=("log",),
            )

        # 纯文本日志
        if "line" in payload:
            return NormalizedFailure(
                id=record.id or f"norm_{int(record.timestamp.timestamp())}",
                timestamp=record.timestamp,
                failure_type="log_pattern",
                severity="error" if "ERROR" in payload["line"] else "warning",
                message=payload["line"][:500],
                source=FailureSource.LOG,
                tags=("log", "text_pattern"),
            )

        return None