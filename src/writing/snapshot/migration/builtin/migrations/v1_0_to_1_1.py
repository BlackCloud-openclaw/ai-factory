# src/writing/snapshot/migration/builtin/migrations/v1_0_to_1_1.py

from ...raw_snapshot import RawSnapshot
from ...version import MigrationContext, MigrationEdge, SchemaVersion


def upcaster_1_0_to_1_1(snapshot: RawSnapshot, ctx: MigrationContext) -> RawSnapshot:
    """将 schema_version 从 1.0 升级到 1.1。"""
    data = dict(snapshot.to_mapping())
    return RawSnapshot.from_mapping(
        schema_version=SchemaVersion(1, 1),
        data=data,
    )


def register_v1_0_to_1_1(registry):
    registry.register_edge(
        MigrationEdge(
            from_version=SchemaVersion(1, 0),
            to_version=SchemaVersion(1, 1),
            upcaster=upcaster_1_0_to_1_1,
        )
    )