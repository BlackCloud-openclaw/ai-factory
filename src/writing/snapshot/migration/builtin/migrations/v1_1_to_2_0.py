# src/writing/snapshot/migration/builtin/migrations/v1_1_to_2_0.py

from ...raw_snapshot import RawSnapshot
from ...version import MigrationContext, MigrationEdge, SchemaVersion


def upcaster_1_1_to_2_0(snapshot: RawSnapshot, ctx: MigrationContext) -> RawSnapshot:
    """将 schema_version 从 1.1 升级到 2.0。"""
    data = dict(snapshot.to_mapping())
    return RawSnapshot.from_mapping(
        schema_version=SchemaVersion(2, 0),
        data=data,
    )


def register_v1_1_to_2_0(registry):
    registry.register_edge(
        MigrationEdge(
            from_version=SchemaVersion(1, 1),
            to_version=SchemaVersion(2, 0),
            upcaster=upcaster_1_1_to_2_0,
        )
    )