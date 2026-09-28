# src/writing/runtime/services.py
"""
Phase 11.2.4: RuntimeServices — 封装 Capability 访问
"""

from typing import Any, Optional, ContextManager

from src.capabilities.runtime import FrozenRuntimeCapabilityRegistry
from .protocols import AuditService

# ========== Phase 15.7-A: Rewriter 导入 ==========
from src.writing.shadow.runner import Rewriter


class RuntimeServices:
    """
    Runtime 服务访问层。

    业务代码通过此层获取各种运行时服务，而不是直接操作 CapabilityRegistry。
    所有服务返回接口（Protocol），隐藏具体实现。
    """

    def __init__(
        self,
        capabilities: FrozenRuntimeCapabilityRegistry,
        rewriter: Optional[Rewriter] = None,  # Phase 15.7-A
    ):
        self._capabilities = capabilities
        self._rewriter = rewriter  # Phase 15.7-A

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
    
    # ========== Phase 15.7-A: Rewriter getter ==========
    @property  # ← 关键：将方法变为属性
    def rewriter(self) -> Optional[Rewriter]:
        """获取 Rewriter 实例（Phase 15.7-A）。"""
        import logging
        logger = logging.getLogger(__name__)
        logger.critical(
            "[15.7-A] RuntimeServices.rewriter property called, returning type=%s",
            type(self._rewriter).__name__ if self._rewriter is not None else "None"
        )
        return self._rewriter