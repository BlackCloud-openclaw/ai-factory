# src/capabilities/__init__.py

# Phase 8: Surface Capability
from .spec import CapabilitySpec, CapabilityMetadata
from .reference import CapabilityRef
from .implementation import CapabilityImplementation
from .registry import CapabilityRegistry
from .protocol import CapabilityLookup
from .errors import (
    CapabilityError,
    CapabilityNotFoundError,
    CapabilityVersionError,
    CapabilityImplementationError,
    CapabilityExecutionError,
)

# Phase 7 遗留常量
from .ids import Matchers, Metrics, Repairs, Triggers

# Phase 11.1: Audit Capability
from .audit import (
    AUDIT_COORDINATOR_ID,
    AUDIT_CAPABILITY_VERSION,
    AUDIT_COORDINATOR_SPEC,
    AuditCapability,
    AuditCapabilityAdapter,
)

# Phase 11.2.1: Runtime Capability
from .runtime import (
    RuntimeCapability,
    RuntimeCapabilityRegistry,
    FrozenRuntimeCapabilityRegistry,
)

# Phase 11.2.3: Snapshot Runtime Capability
from .runtime.snapshot import (
    SNAPSHOT_REPOSITORY_ID,
    SNAPSHOT_REPOSITORY_VERSION,
    SNAPSHOT_REPOSITORY_SPEC,
    SNAPSHOT_VERSION_STORE_ID,
    SNAPSHOT_VERSION_STORE_VERSION,
    SNAPSHOT_VERSION_STORE_SPEC,
    SNAPSHOT_TRANSPORT_ID,
    SNAPSHOT_TRANSPORT_VERSION,
    SNAPSHOT_TRANSPORT_SPEC,
    SnapshotRepositoryCapability,
    SnapshotVersionStoreCapability,
    SnapshotTransportCapability,
    SnapshotCapabilityAdapter,
)

__all__ = [
    # Phase 8 核心
    "CapabilitySpec",
    "CapabilityMetadata",
    "CapabilityRef",
    "CapabilityImplementation",
    "CapabilityRegistry",
    "CapabilityLookup",
    "CapabilityError",
    "CapabilityNotFoundError",
    "CapabilityVersionError",
    "CapabilityImplementationError",
    "CapabilityExecutionError",
    # Phase 7 遗留
    "Matchers",
    "Metrics",
    "Repairs",
    "Triggers",
    # Phase 11.1
    "AUDIT_COORDINATOR_ID",
    "AUDIT_CAPABILITY_VERSION",
    "AUDIT_COORDINATOR_SPEC",
    "AuditCapability",
    "AuditCapabilityAdapter",
    # Phase 11.2.1
    "RuntimeCapability",
    "RuntimeCapabilityRegistry",
    "FrozenRuntimeCapabilityRegistry",
    # Phase 11.2.3
    "SNAPSHOT_REPOSITORY_ID",
    "SNAPSHOT_REPOSITORY_VERSION",
    "SNAPSHOT_REPOSITORY_SPEC",
    "SNAPSHOT_VERSION_STORE_ID",
    "SNAPSHOT_VERSION_STORE_VERSION",
    "SNAPSHOT_VERSION_STORE_SPEC",
    "SNAPSHOT_TRANSPORT_ID",
    "SNAPSHOT_TRANSPORT_VERSION",
    "SNAPSHOT_TRANSPORT_SPEC",
    "SnapshotRepositoryCapability",
    "SnapshotVersionStoreCapability",
    "SnapshotTransportCapability",
    "SnapshotCapabilityAdapter",
]