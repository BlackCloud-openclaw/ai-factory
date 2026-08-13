# src/writing/projection_repository.py

import json
import asyncpg
import os
from pathlib import Path
from typing import Optional, Protocol
from src.writing.narrative_projection import NarrativeProjection


class ProjectionRepository(Protocol):
    """投影存储协议，定义存储接口"""
    def save(self, projection: NarrativeProjection) -> None:
        ...
    def load(self, chapter_id: str) -> Optional[NarrativeProjection]:
        ...
    def latest(self) -> Optional[NarrativeProjection]:
        ...


class FileProjectionRepository:
    """基于文件的投影存储"""
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, chapter_id: str) -> Path:
        return self.base_dir / f"{chapter_id}.json"

    def _chapter_number(self, path: Path) -> int:
        name = path.stem
        try:
            return int(name.split("_")[1])
        except (IndexError, ValueError):
            return 0

    def save(self, projection: NarrativeProjection) -> None:
        path = self._get_path(projection.chapter_id)
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(projection.to_dict(), f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(path)

    def load(self, chapter_id: str) -> Optional[NarrativeProjection]:
        path = self._get_path(chapter_id)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return NarrativeProjection.from_dict(data)

    def latest(self) -> Optional[NarrativeProjection]:
        files = sorted(self.base_dir.glob("chapter_*.json"), key=self._chapter_number)
        if not files:
            return None
        latest_file = files[-1]
        with open(latest_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return NarrativeProjection.from_dict(data)


class PostgresProjectionRepository:
    """基于 PostgreSQL 的投影存储，支持事务连接"""

    def __init__(self, conn: Optional[asyncpg.Connection] = None):
        self._conn = conn

    async def save_with_conn(self, projection: NarrativeProjection, conn: asyncpg.Connection) -> None:
        import json
        await conn.execute("""
            INSERT INTO narrative_projection_snapshots
            (id, novel_id, chapter, event_id, projection_data, created_at)
            VALUES ($1, $2, $3, $4, $5, NOW())
            ON CONFLICT (id) DO UPDATE SET
                projection_data = EXCLUDED.projection_data,
                created_at = NOW()
        """, projection.projection_id, "default_novel", 1, 0,
           json.dumps(projection.to_dict()))

    async def latest_with_conn(self, conn: asyncpg.Connection) -> Optional[NarrativeProjection]:
        import json
        row = await conn.fetchrow("""
            SELECT projection_data
            FROM narrative_projection_snapshots
            ORDER BY created_at DESC LIMIT 1
        """)
        if not row:
            return None
        return NarrativeProjection.from_dict(json.loads(row["projection_data"]))

    async def load_with_conn(self, projection_id: str, conn: asyncpg.Connection) -> Optional[NarrativeProjection]:
        import json
        row = await conn.fetchrow(
            "SELECT projection_data FROM narrative_projection_snapshots WHERE id = $1",
            projection_id
        )
        if not row:
            return None
        return NarrativeProjection.from_dict(json.loads(row["projection_data"]))    