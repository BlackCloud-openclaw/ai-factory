# src/capabilities/implementation.py

from typing import Protocol, Dict, Any, Sequence, runtime_checkable


@runtime_checkable
class CapabilityImplementation(Protocol):
    """
    Capability 执行接口

    所有 Capability 实现必须：
    1. 无状态（Stateless）
    2. 实现 match 方法
    3. 可安全地在多线程/多进程环境中调用
    """

    def match(self, text: str, config: Dict[str, Any]) -> Sequence[Dict[str, Any]]:
        """
        在文本中匹配 Capability

        Args:
            text: 待匹配的文本
            config: 匹配配置（由 Surface 提供）

        Returns:
            匹配结果序列，每个结果包含：
            - start: 起始位置
            - end: 结束位置
            - text: 匹配文本
            - pattern_type: 模式类型标识
            - 其他自定义字段
        """
        ...