# src/writing/snapshot/migration/builtin/versions/v2_0.py

from ...version import SchemaVersion, VersionNode, VersionType


def register_v2_0(registry):
    registry.register_node(
        VersionNode(version=SchemaVersion(2, 0), version_type=VersionType.MAJOR)
    )