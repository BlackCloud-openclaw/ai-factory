from typing import Optional
from src.writing.audit import AuditCoordinator, AuditConfig
from src.writing.audit.payload_resolver import PayloadResolver, MemoryPayloadResolver
from .implementation import AuditCapability


class AuditCapabilityAdapter:
    @staticmethod
    def create(
        resolver: Optional[PayloadResolver] = None,
        config: Optional[AuditConfig] = None,
    ) -> AuditCapability:
        if resolver is None:
            resolver = MemoryPayloadResolver()
        coordinator = AuditCoordinator(resolver=resolver, config=config)
        return AuditCapability(coordinator)