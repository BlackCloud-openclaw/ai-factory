"""
Phase 13.1: Planner Output Contract

Planner 输出叙事意图 + 执行约束，而非混在一起。
两者职责分离，不存在冲突。
"""

from pydantic import BaseModel, Field
from src.writing.planning_contract import PlanningContract
from src.writing.narrative_intent import NarrativeIntent


class PlannerOutput(BaseModel):
    """
    Planner 的完整输出。

    上层是 NarrativeIntent（叙事控制层），
    下层是 ExecutionContract（Runtime执行层）。
    两者职责分离，不存在冲突。
    """

    narrative_intent: NarrativeIntent = Field(
        ...,
        description="叙事控制意图（为什么写、改变什么）"
    )
    execution_contract: PlanningContract = Field(
        ...,
        description="Runtime 执行约束（写什么、在哪写、谁参与）"
    )