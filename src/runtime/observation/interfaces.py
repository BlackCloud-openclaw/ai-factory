# src/runtime/observation/interfaces.py

"""
Phase 8 Observation Protocol

IntentCompiler 依赖此协议，而非重新定义 ObservationReader。
"""

from typing import List, Protocol, Tuple, runtime_checkable


@runtime_checkable
class ObservationProtocol(Protocol):
    """
    Phase 8 Observation 协议

    这是 Runtime 对 Narrative 层的稳定输出契约。
    """

    def get_dimension(self, name: str) -> float | None:
        """获取特定维度的观察值（如 dialogue_ratio, transition_score）"""
        ...

    def get_evidence(self, dimension: str) -> List[str]:
        """获取特定维度的证据文本"""
        ...

    def get_all_dimensions(self) -> Tuple[str, ...]:
        """获取所有维度名称"""
        ...