from packaging.version import Version
from src.capabilities.spec import CapabilitySpec, CapabilityMetadata

AUDIT_COORDINATOR_ID = "builtin.runtime.audit.coordinator"
AUDIT_CAPABILITY_VERSION = "1.0"

AUDIT_COORDINATOR_SPEC = CapabilitySpec(
    id=AUDIT_COORDINATOR_ID,
    version=Version(AUDIT_CAPABILITY_VERSION),
    metadata=CapabilityMetadata(
        display_name="Audit Coordinator",
        description="Writer runtime audit coordinator",
        tags=("audit", "runtime"),
    ),
)