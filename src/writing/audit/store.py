# src/writing/audit/store.py
"""
Phase 10.3.3: AuditReportStore — 报告持久化与查询
"""

import json
import logging
from pathlib import Path
from typing import Optional, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from .reporter import ComprehensiveReport


@dataclass(frozen=True)
class ReportEntry:
    """报告条目（轻量元数据）。"""
    execution_id: str
    novel_id: str
    volume: int
    chapter: int
    scene_idx: int
    created_at: datetime
    summary: dict
    report_path: Path


class AuditReportStore:
    """
    审计报告存储。

    ComprehensiveReport 保存到文件系统，支持查询。
    不缓存完整报告对象，由调用方按需加载。
    """

    def __init__(self, base_dir: Path):
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._base_dir / "index.json"
        self._entries: list[ReportEntry] = []
        self._load_index()

    def save(self, report: ComprehensiveReport, novel_id: Optional[str] = None) -> Path:
        """保存报告。"""
        novel_id = novel_id or report.preservation.novel_id or "unknown"
        timestamp = datetime.now(timezone.utc).isoformat()
        filename = f"{report.execution_id}_{timestamp}.json"
        path = self._base_dir / novel_id / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        data = report.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        entry = ReportEntry(
            execution_id=report.execution_id,
            novel_id=novel_id,
            volume=getattr(report.preservation, "volume", 0),
            chapter=getattr(report.preservation, "chapter", 0),
            scene_idx=getattr(report.preservation, "scene_idx", 0),
            created_at=datetime.now(timezone.utc),
            summary=report.summary,
            report_path=path,
        )
        self._entries.append(entry)
        self._save_index()
        return path

    def list(self, novel_id: Optional[str] = None, limit: int = 50) -> Sequence[ReportEntry]:
        """列出报告条目（轻量元数据）。"""
        entries = self._entries
        if novel_id:
            entries = [e for e in entries if e.novel_id == novel_id]
        entries = sorted(entries, key=lambda x: x.created_at, reverse=True)
        return entries[:limit]

    def load(self, entry: ReportEntry) -> Optional[ComprehensiveReport]:
        """从文件加载完整报告。"""
        if not entry.report_path.exists():
            return None
        try:
            with open(entry.report_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ComprehensiveReport.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logging.getLogger("audit.store").warning(f"Failed to load report {entry.execution_id}: {e}")
            return None

    def load_by_id(self, execution_id: str) -> Optional[ComprehensiveReport]:
        """通过 execution_id 加载报告。"""
        for entry in self._entries:
            if entry.execution_id == execution_id:
                return self.load(entry)
        return None

    def _save_index(self) -> None:
        data = [
            {
                "execution_id": e.execution_id,
                "novel_id": e.novel_id,
                "volume": e.volume,
                "chapter": e.chapter,
                "scene_idx": e.scene_idx,
                "created_at": e.created_at.isoformat(),
                "summary": e.summary,
                "report_path": str(e.report_path),
            }
            for e in self._entries
        ]
        with open(self._index_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_index(self) -> None:
        if not self._index_path.exists():
            return
        try:
            with open(self._index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._entries = [
                ReportEntry(
                    execution_id=e["execution_id"],
                    novel_id=e["novel_id"],
                    volume=e["volume"],
                    chapter=e["chapter"],
                    scene_idx=e["scene_idx"],
                    created_at=datetime.fromisoformat(e["created_at"]),
                    summary=e["summary"],
                    report_path=Path(e["report_path"]),
                )
                for e in data
            ]
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logging.getLogger("audit.store").warning(f"Failed to load index: {e}")
            self._entries = []