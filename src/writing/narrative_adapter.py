# src/writing/narrative_adapter.py

"""
Phase 13.1: NarrativeAwareWriterAdapter

在不破坏 Phase 10 Runtime ABI 的前提下，
将 NarrativeIntent 注入 Writer 执行过程。
"""

from typing import Optional
from src.writing.controlled_writer import ControlledWriter
from src.writing.planner_output import PlannerOutput
from src.writing.narrative_intent import NarrativeIntent
from src.writing.planning_contract import PlanningContract


class NarrativeAwareWriterAdapter:
    """
    适配器：为 Writer Runtime 注入叙事控制层。

    保持 Writer.execute(PlanningContract) 签名不变。
    """

    def __init__(self, writer: ControlledWriter):
        self._writer = writer

    async def execute(
        self,
        planner_output: PlannerOutput,
        context: Optional[dict] = None,
    ):
        """
        执行 Writer，并传递 NarrativeIntent 作为上下文。

        当前实现：将 narrative_intent 序列化到 context 中，
        供 Writer 内部消费（Phase 13.2 将扩展）。
        """
        execution_contract = planner_output.execution_contract
        narrative_intent = planner_output.narrative_intent

        # 将 intent 注入 context（不修改 Writer 签名）
        if context is None:
            context = {}
        context["narrative_intent"] = narrative_intent.to_dict()

        # 调用原有 Writer.execute，保持 ABI 不变
        return await self._writer.execute(execution_contract, context=context)

    async def execute_with_intent(
        self,
        execution_contract: PlanningContract,
        narrative_intent: NarrativeIntent,
        context: Optional[dict] = None,
    ):
        """
        显式传递 NarrativeIntent 的便捷方法。
        """
        if context is None:
            context = {}
        context["narrative_intent"] = narrative_intent.to_dict()
        return await self._writer.execute(execution_contract, context=context)