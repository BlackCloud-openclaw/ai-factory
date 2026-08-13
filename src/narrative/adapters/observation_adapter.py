# src/narrative/adapters/observation_adapter.py

"""
Observation Adapter — 将 Phase 8 Observation 适配为 IntentCompiler 输入

位置：src/narrative/adapters/（不是 compiler）
职责：将 Runtime 数据结构转换为 Narrative 可消费的协议

注意：当前版本接收 dict 作为输入，这是 Phase 8 ObservationCompiler
当前输出格式的临时适配。生产环境最终应直接接收 Observation IR 对象。
"""

from typing import List, Optional, Tuple

from src.runtime.observation.interfaces import ObservationProtocol


class ObservationAdapter(ObservationProtocol):
    """
    Phase 8 Observation 适配器

    将 Phase 8 ObservationCompiler 的输出适配为 IntentCompiler 可读的 ObservationProtocol。

    TEMPORARY: 当前接收 dict，这是对 Phase 8 当前输出格式的适配。
    待 Phase 8 Observation IR 稳定后，应直接接收 Observation 对象。
    """

    def __init__(self, data: dict):
        """
        Args:
            data: Phase 8 ObservationCompiler 的输出（当前为 dict 格式）
        """
        self._data = data

    def get_dimension(self, name: str) -> Optional[float]:
        """获取特定维度的观察值"""
        if "dimensions" in self._data:
            return self._data["dimensions"].get(name)
        return self._data.get(name)

    def get_evidence(self, dimension: str) -> List[str]:
        """获取特定维度的证据文本"""
        if "evidence" in self._data:
            return self._data["evidence"].get(dimension, [])
        return []

    def get_all_dimensions(self) -> Tuple[str, ...]:
        """获取所有维度名称"""
        if "dimensions" in self._data:
            return tuple(self._data["dimensions"].keys())
        return tuple(self._data.keys())