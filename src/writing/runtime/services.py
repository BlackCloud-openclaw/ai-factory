# src/writing/runtime/services.py
"""
Phase 11.2.4: RuntimeServices — 封装 Capability 访问
"""

from typing import Any, Optional, ContextManager

from src.capabilities.runtime import FrozenRuntimeCapabilityRegistry
from .protocols import AuditService


class RuntimeServices:
    """
    Runtime 服务访问层。

    业务代码通过此层获取各种运行时服务，而不是直接操作 CapabilityRegistry。
    所有服务返回接口（Protocol），隐藏具体实现。
    """

    def __init__(self, capabilities: FrozenRuntimeCapabilityRegistry):
        self._capabilities = capabilities

    def audit(self) -> AuditService:
        """
        获取审计服务。

        Returns:
            AuditService 协议实例
        """
        capability = self._capabilities.require("builtin.runtime.audit.coordinator")
        return capability.get()

    def audit_context(
        self,
        novel_id: str,
        volume: int,
        chapter: int,
        scene_idx: int,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ContextManager:
        """
        直接获取审计上下文（简化调用）。
        """
        service = self.audit()
        return service.audit(novel_id, volume, chapter, scene_idx, metadata=metadata)