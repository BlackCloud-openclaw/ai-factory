from dataclasses import dataclass
from .enforcement_mode import EnforcementMode

@dataclass(frozen=True)
class ValidationPolicy:
    # 现有字段（默认值保持不变）
    allow_degraded_pass: bool = False
    max_retry: int = 3
    fail_on_error: bool = True
    recovery_enabled: bool = False

    # 新增字段
    enforcement_mode: EnforcementMode = EnforcementMode.OBSERVE

    @classmethod
    def development(cls) -> "ValidationPolicy":
        """开发环境默认策略：允许降级、OBSERVE 模式"""
        return cls(
            allow_degraded_pass=True,
            max_retry=3,
            fail_on_error=False,
            enforcement_mode=EnforcementMode.OBSERVE
        )

    @classmethod
    def production(cls) -> "ValidationPolicy":
        """生产环境默认策略：禁止降级、RETRY 模式"""
        return cls(
            allow_degraded_pass=False,
            max_retry=3,
            enforcement_mode=EnforcementMode.RETRY
        )