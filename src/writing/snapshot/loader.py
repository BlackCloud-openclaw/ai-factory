# src/writing/snapshot/loader.py
"""
Snapshot Loader — 支持自动版本迁移（B2）
"""

from pathlib import Path
from typing import Optional

from .decoder import SnapshotDecoder
from .migration import (
    CurrentSchemaProvider,
    MigrationContextFactory,
    RawSnapshot,
    SchemaVersion,
    SnapshotMigrator,
    SnapshotVersionTooNewError,
)
from .models import PipelineSnapshot
from .serializer import JsonSerializer, Serializer


class SnapshotLoader:
    """
    快照加载器，支持从旧版本自动迁移。

    依赖三个 Protocol（依赖注入）：
    - SnapshotMigrator: 执行版本迁移
    - CurrentSchemaProvider: 提供当前支持的版本
    - MigrationContextFactory: 创建迁移上下文
    """

    def __init__(
        self,
        migrator: SnapshotMigrator,
        schema_provider: CurrentSchemaProvider,
        context_factory: MigrationContextFactory,
        serializer: Optional[Serializer] = None,
    ):
        self._migrator = migrator
        self._schema_provider = schema_provider
        self._context_factory = context_factory
        self._serializer = serializer or JsonSerializer()

    def load(self, path: Path) -> PipelineSnapshot:
        data = path.read_bytes()
        raw = self._deserialize_to_raw(data)

        target = self._schema_provider.get()

        if raw.schema_version == target:
            return self._decode(raw)

        if raw.schema_version.is_newer_than(target):
            raise SnapshotVersionTooNewError(
                f"Snapshot version {raw.schema_version} is newer than "
                f"current supported version {target}"
            )

        context = self._context_factory.create()
        raw = self._migrator.migrate(raw, target, context)
        return self._decode(raw)

    def _deserialize_to_raw(self, data: bytes) -> RawSnapshot:
        snapshot = self._serializer.deserialize(data)
        raw_data = snapshot.model_dump()
        schema_version = SchemaVersion.parse(snapshot.manifest.schema_version)
        return RawSnapshot.from_mapping(
            schema_version=schema_version,
            data=raw_data,
        )

    def _decode(self, raw: RawSnapshot) -> PipelineSnapshot:
        decoder = SnapshotDecoder()
        return decoder.decode(raw.to_mapping())