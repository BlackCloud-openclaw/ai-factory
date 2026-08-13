# src/writing/snapshot/migration/__init__.py

# B1.1
from .version import (
    CapabilityId,
    Clock,
    FixedClock,
    FixedRandom,
    MigrationContext,
    MigrationEdge,
    RandomProvider,
    SchemaVersion,
    Upcaster,
    VersionNode,
    VersionType,
    freeze_mapping,
)

# B1.2
from .deep_freeze import deep_freeze
from .raw_snapshot import RawSnapshot

# B1.3
from .graph import MigrationGraph, PathStrategy

# B1.4
from .errors import (
    MigrationError,
    MigrationExecutionError,
    MigrationPathNotFoundError,
    SnapshotVersionTooNewError,
)
from .runtime import MigrationRuntime, MigrationObserver

# B2 - Runtime Configuration
from .schema import CURRENT_SCHEMA_VERSION, MINIMUM_SUPPORTED_VERSION

# B2 - Builtin Migrations
from .builtin import register_builtin_migrations

# B2 - Registry
from .registry import MigrationRegistry

# B2 - Protocols
from .migrator import SnapshotMigrator
from .schema_provider import CurrentSchemaProvider, StaticSchemaProvider
from .context_factory import MigrationContextFactory, DefaultMigrationContextFactory


__all__ = [
    # B1.1
    "SchemaVersion",
    "CapabilityId",
    "VersionType",
    "VersionNode",
    "MigrationEdge",
    "MigrationContext",
    "Clock",
    "RandomProvider",
    "FixedClock",
    "FixedRandom",
    "Upcaster",
    "freeze_mapping",
    # B1.2
    "deep_freeze",
    "RawSnapshot",
    # B1.3
    "MigrationGraph",
    "PathStrategy",
    # B1.4
    "MigrationRuntime",
    "MigrationObserver",
    "MigrationError",
    "MigrationPathNotFoundError",
    "MigrationExecutionError",
    "SnapshotVersionTooNewError",
    # B2
    "CurrentSchemaProvider",
    "StaticSchemaProvider",
    "MigrationContextFactory",
    "DefaultMigrationContextFactory",
    "SnapshotMigrator",
    "MigrationRegistry",
    "CURRENT_SCHEMA_VERSION",
    "MINIMUM_SUPPORTED_VERSION",
    "register_builtin_migrations",
]