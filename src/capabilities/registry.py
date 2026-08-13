# src/capabilities/registry.py

from typing import Dict, Optional, Tuple
from types import MappingProxyType

from src.capabilities.spec import CapabilitySpec
from src.capabilities.reference import CapabilityRef
from src.capabilities.implementation import CapabilityImplementation
from src.capabilities.errors import (
    CapabilityNotFoundError,
    CapabilityVersionError,
    CapabilityImplementationError,
)


class CapabilityRegistry:
    """
    Capability 只读目录

    设计决策：
    - 单版本活动模型：每个 Capability ID 只有一个活动版本
    - 由 Composition Root 创建，完全不可变
    - 所有公开状态均为 MappingProxyType

    Phase 8 不实现多版本共存，留待 Phase 9。
    """

    def __init__(
        self,
        specs: Dict[str, CapabilitySpec],
        implementations: Dict[str, CapabilityImplementation],
    ):
        """
        Args:
            specs: {capability_id: CapabilitySpec}
            implementations: {capability_id: CapabilityImplementation}
        """
        # 验证每个 Spec 都有对应的 Implementation
        for id in specs:
            if id not in implementations:
                raise CapabilityImplementationError(
                    f"Capability '{id}' has Spec but no Implementation"
                )

        # 使用 MappingProxyType 确保不可变
        self._specs = MappingProxyType(dict(specs))
        self._implementations = MappingProxyType(dict(implementations))

    def get_spec(self, ref: CapabilityRef) -> CapabilitySpec:
        """获取 CapabilitySpec，不存在时抛出异常"""
        spec = self._specs.get(ref.id)
        if spec is None:
            raise CapabilityNotFoundError(str(ref))

        # 如果指定了版本，校验是否匹配
        if ref.version is not None and spec.version != ref.version:
            raise CapabilityVersionError(str(ref), str(spec.version))

        return spec

    def get_impl(self, ref: CapabilityRef) -> CapabilityImplementation:
        """
        获取 CapabilityImplementation，不存在时抛出异常

        注意：不依赖 get_spec()，版本检查由 Builder 负责。
        """
        impl = self._implementations.get(ref.id)
        if impl is None:
            raise CapabilityImplementationError(
                f"Implementation not found for capability '{ref.id}'"
            )

        # 验证 Implementation 符合 Protocol
        if not isinstance(impl, CapabilityImplementation):
            raise CapabilityImplementationError(
                f"Implementation for '{ref.id}' does not implement CapabilityImplementation Protocol"
            )

        return impl

    def get_all_specs(self) -> Tuple[CapabilitySpec, ...]:
        """返回所有 CapabilitySpec（只读）"""
        return tuple(self._specs.values())

    def get_all_ids(self) -> Tuple[str, ...]:
        """返回所有 Capability ID"""
        return tuple(self._specs.keys())

    def find_spec(self, ref: CapabilityRef) -> Optional[CapabilitySpec]:
        """查找 CapabilitySpec，返回 Optional（不抛出）"""
        try:
            return self.get_spec(ref)
        except (CapabilityNotFoundError, CapabilityVersionError):
            return None

    def has(self, ref: CapabilityRef) -> bool:
        """检查 Capability 是否存在（仅检查 id，忽略版本）"""
        return ref.id in self._specs

    def list_specs(self) -> Tuple[CapabilitySpec, ...]:
        """列出所有 Spec（只读，用于 Builder）"""
        return tuple(self._specs.values())

    def list_impls(self) -> Tuple[CapabilityImplementation, ...]:
        """列出所有 Implementation（只读，用于 Builder）"""
        return tuple(self._implementations.values())