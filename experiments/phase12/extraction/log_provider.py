"""
LogFailureProvider：从日志文件读取失败记录（无状态）
"""

import re
import json
from pathlib import Path
from datetime import datetime
from typing import Iterator, List, Optional, Any, Dict

from .models import RawFailureRecord, FailureSource
from .provider import FailureProvider


class LogFailureProvider:
    """从日志文件读取失败记录（流式，无状态）"""

    DEFAULT_PATTERNS = [
        r"Low compliance",
        r"Runtime validation failed",
        r"Validation failed",
        r"Scene completion failed",
        r"ERROR.*validation",
        r"WARNING.*Low compliance",
    ]

    def __init__(
        self,
        log_paths: List[Path],
        failure_patterns: Optional[List[str]] = None,
        max_records: int = 0,  # 0 = 无限制
    ):
        self._log_paths = log_paths
        self._patterns = [re.compile(p) for p in (failure_patterns or self.DEFAULT_PATTERNS)]
        self._max_records = max_records

    def iter_records(self) -> Iterator[RawFailureRecord]:
        """流式迭代失败记录（无状态，可多次调用）"""
        count = 0
        for path in self._log_paths:
            if not path.exists():
                continue
            for record in self._parse_file(path, count):
                count += 1
                yield record
                if self._max_records > 0 and count >= self._max_records:
                    return

    def _parse_file(self, path: Path, start_count: int) -> Iterator[RawFailureRecord]:
        count = start_count
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                # 尝试解析 JSON 行
                if line.startswith("{") and line.endswith("}"):
                    try:
                        data = json.loads(line)
                        if self._is_failure(data):
                            count += 1
                            yield RawFailureRecord(
                                source=FailureSource.LOG,
                                payload=data,
                                timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
                                id=data.get("id", f"log_{count}"),
                            )
                    except json.JSONDecodeError:
                        pass
                    continue

                # 文本日志匹配
                if any(p.search(line) for p in self._patterns):
                    count += 1
                    yield RawFailureRecord(
                        source=FailureSource.LOG,
                        payload={"line": line.strip()},
                        timestamp=datetime.now(),
                        id=f"log_{count}",
                    )

                if self._max_records > 0 and count >= self._max_records:
                    return

    def _is_failure(self, data: Dict[str, Any]) -> bool:
        level = data.get("level", "").upper()
        if level in ("ERROR", "WARNING"):
            return True
        message = data.get("message", "")
        return any(p.search(message) for p in self._patterns)