# src/capabilities/errors.py

class CapabilityError(Exception):
    """所有 Capability 异常的基类"""
    pass


class CapabilityNotFoundError(CapabilityError):
    """Capability ID 不存在"""

    def __init__(self, ref: str):
        super().__init__(f"Capability not found: {ref}")
        self.ref = ref


class CapabilityVersionError(CapabilityError):
    """版本不匹配或不可用"""

    def __init__(self, ref: str, available: str):
        super().__init__(f"Capability version mismatch: requested {ref}, available {available}")
        self.ref = ref
        self.available = available


class CapabilityImplementationError(CapabilityError):
    """Implementation 注册或调用失败"""
    pass


class CapabilityExecutionError(CapabilityError):
    """Capability 执行失败（插件边界异常）"""
    pass