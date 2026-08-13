"""
Phase 13.2: NarrativeProjectionService

提供 Projection 的加载和保存服务，解耦 Orchestrator 与 Repository。
"""

import asyncpg
from typing import Optional
from pathlib import Path
from src.writing.narrative_projection import NarrativeProjection
from src.writing.projection_repository import (
    ProjectionRepository,
    FileProjectionRepository,
    PostgresProjectionRepository,
)


class NarrativeProjectionService:
    """叙事投影服务 - Runtime 边界层"""

    def __init__(
        self,
        repository: Optional[ProjectionRepository] = None,
        conn: Optional[asyncpg.Connection] = None,
    ):
        """支持传入数据库连接以实现事务一致性"""
        self._conn = conn
        if repository is not None:
            self._repository = repository
        else:
            if conn is not None:
                self._repository = PostgresProjectionRepository(conn)
            else:
                self._repository = FileProjectionRepository(
                    Path("experiments/narrative/projections")
                )

    def load_current(self) -> Optional[NarrativeProjection]:
        if self._conn is not None and hasattr(self._repository, 'latest_with_conn'):
            return self._repository.latest_with_conn(self._conn)
        return self._repository.latest()

    def save(self, projection: NarrativeProjection) -> None:
        if self._conn is not None and hasattr(self._repository, 'save_with_conn'):
            return self._repository.save_with_conn(projection, self._conn)
        self._repository.save(projection)

    def load_by_chapter(self, chapter_id: str) -> Optional[NarrativeProjection]:
        if self._conn is not None and hasattr(self._repository, 'load_with_conn'):
            return self._repository.load_with_conn(chapter_id, self._conn)
        return self._repository.load(chapter_id)