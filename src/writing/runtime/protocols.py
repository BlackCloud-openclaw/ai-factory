# src/writing/runtime/protocols.py
"""
Phase 11.2.4: Runtime Service Protocols
"""

from typing import Protocol, Any, Optional, ContextManager


class AuditService(Protocol):
    """
    审计服务协议。

    业务层通过此接口获取审计能力，不直接依赖 AuditCoordinator。
    """

    def audit(
        self,
        novel_id: str,
        volume: int,
        chapter: int,
        scene_idx: int,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ContextManager:
        """
        获取审计上下文管理器。

        Returns:
            审计上下文（支持 with 语句）
        """
        ...

    def audit_context(
        self,
        novel_id: str,
        volume: int,
        chapter: int,
        scene_idx: int,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ContextManager:
        """
        audit() 的别名，用于 with 语句。

        Returns:
            审计上下文
        """
        ...