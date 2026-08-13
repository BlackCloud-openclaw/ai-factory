# src/writing/snapshot/migration/builtin/versions/v1_1.py

from ...version import SchemaVersion, VersionNode, VersionType


def register_v1_1(registry):
    registry.register_node(
        VersionNode(version=SchemaVersion(1, 1), version_type=VersionType.MINOR)
    )