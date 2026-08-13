# src/capabilities/runtime/protocol.py
"""
Phase 11.2.1: RuntimeCapability Protocol

Runtime Capability 与 Surface Capability 的区别：
    - Surface Capability: match(text, config) -> matches
    - Runtime Capability: get() -> Any (runtime service object)
"""

from typing import Protocol, Any, runtime_checkable


@runtime_checkable
class RuntimeCapability(Protocol):
    """
    Runtime Service Capability.

    所有 Runtime 能力必须提供 get() 方法，
    返回具体的运行时服务对象。
    """

    def get(self) -> Any:
        """
        获取运行时服务对象。

        Returns:
            具体的服务实例（如 AuditCoordinator, SnapshotStore 等）
        """
        ...