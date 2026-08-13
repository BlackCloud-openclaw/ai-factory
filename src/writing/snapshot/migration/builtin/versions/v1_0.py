# src/writing/snapshot/migration/builtin/versions/v1_0.py

from ...version import SchemaVersion, VersionNode, VersionType


def register_v1_0(registry):
    registry.register_node(
        VersionNode(version=SchemaVersion(1, 0), version_type=VersionType.MINOR)
    )