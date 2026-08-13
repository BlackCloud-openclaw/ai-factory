from enum import Enum

class EnforcementMode(Enum):
    """控制 Contract Retry 的执行策略"""
    OBSERVE = "observe"   # 仅记录缺失，不触发重试
    RETRY = "retry"       # 允许触发重试
    STRICT = "strict"     # 重试耗尽后硬失败（后续实现）