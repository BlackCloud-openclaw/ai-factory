# src/writing/bootstrap/composition_root.py
"""
Phase 11.2.4: Composition Root — 全局启动组装
Phase 14.0C-2: 注入 ValidationPolicy
"""

import os
from dataclasses import dataclass

from src.capabilities.runtime import FrozenRuntimeCapabilityRegistry
from src.writing.bootstrap.runtime_capabilities import build_runtime_capabilities
from src.writing.runtime.services import RuntimeServices
from src.writing.runtime import ValidationPolicy


@dataclass(frozen=True)
class WriterRuntime:
    runtime_capabilities: FrozenRuntimeCapabilityRegistry
    runtime_services: RuntimeServices
    validation_policy: ValidationPolicy  # Phase 14.0C-2 新增

    def __post_init__(self):
        required_ids = [
            "builtin.runtime.audit.coordinator",
            "builtin.runtime.snapshot.repository",
            "builtin.runtime.snapshot.version_store",
            "builtin.runtime.snapshot.transport",
        ]
        for cap_id in required_ids:
            if not self.runtime_capabilities.has(cap_id):
                raise RuntimeError(
                    f"Required runtime capability '{cap_id}' not registered. "
                    "Ensure build_runtime_capabilities() includes all required capabilities."
                )


def build_writer_runtime() -> WriterRuntime:
    """
    构建完整的 Writer Runtime 环境。
    """
    capabilities = build_runtime_capabilities()
    services = RuntimeServices(capabilities)

    # Phase 14.0C-2: 根据环境变量选择策略
    env = os.getenv("ENVIRONMENT", "development")
    if env == "production":
        validation_policy = ValidationPolicy.production()
    else:
        validation_policy = ValidationPolicy.development()

    return WriterRuntime(
        runtime_capabilities=capabilities,
        runtime_services=services,
        validation_policy=validation_policy,
    )