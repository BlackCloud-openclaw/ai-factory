from src.writing.audit import AuditCoordinator


class AuditCapability:
    def __init__(self, coordinator: AuditCoordinator):
        self._coordinator = coordinator

    def get(self) -> AuditCoordinator:
        return self._coordinator