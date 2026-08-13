"""
Phase 12.2B: SnapshotWriterAdapter

将 ControlledWriter.execute_with_snapshot 的返回 (snapshot, result) 适配为仅返回 snapshot。
"""

from typing import Optional, Any
from src.writing.controlled_writer import ControlledWriter
from src.writing.evaluation import EvaluationSnapshot
from src.writing.planning_contract import PlanningContract


class SnapshotWriterAdapter:
    """
    适配器：使 ControlledWriter 符合 WriterProtocol（仅返回 snapshot）。
    """

    def __init__(self, writer: ControlledWriter):
        self.writer = writer

    async def execute_with_snapshot(
        self,
        contract: PlanningContract,
        scene_before: str = "",
        previous_scene_text: Optional[str] = None,
        character_summary: Optional[dict] = None,
        world_summary: Optional[dict] = None,
        **kwargs,  # 接受额外参数（如 novel_id, volume 等）
    ) -> EvaluationSnapshot:
        """执行 Writer 并返回 snapshot（忽略 result）"""
        snapshot, _ = await self.writer.execute_with_snapshot(
            contract=contract,
            scene_before=scene_before,
            previous_scene_text=previous_scene_text,
            character_summary=character_summary,
            world_summary=world_summary,
        )
        return snapshot
