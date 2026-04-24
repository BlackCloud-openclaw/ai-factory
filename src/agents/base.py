# src/agents/base.py
from typing import Dict, Any
from abc import ABC, abstractmethod

from src.orchestrator.state import AgentState


class BaseAgent(ABC):
    """所有 Agent 必须实现的统一接口"""

    @abstractmethod
    async def run(self, state: AgentState) -> Dict[str, Any]:
        """
        接受当前状态，返回需要更新的字段字典（增量更新）。
        禁止直接修改 state 对象。
        """
        pass
