# src/capabilities/protocol.py

from typing import Protocol, Optional, runtime_checkable

from src.capabilities.spec import CapabilitySpec
from src.capabilities.reference import CapabilityRef
from src.capabilities.implementation import CapabilityImplementation


@runtime_checkable
class CapabilityLookup(Protocol):
    """
    Capability 查找协议 — Runtime 的唯一依赖。

    has() 仅检查 capability id 是否存在，不验证版本。
    版本验证由 get_spec() 完成。
    """

    def get_spec(self, ref: CapabilityRef) -> CapabilitySpec:
        """获取 CapabilitySpec，不存在或版本不匹配时抛出异常"""
        ...

    def get_impl(self, ref: CapabilityRef) -> CapabilityImplementation:
        """获取 CapabilityImplementation，不存在时抛出异常"""
        ...

    def has(self, ref: CapabilityRef) -> bool:
        """检查 Capability 是否存在（仅检查 id，忽略版本）"""
        ...