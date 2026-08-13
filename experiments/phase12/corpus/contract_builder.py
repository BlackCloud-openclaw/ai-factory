"""
Phase 12.2B-2: Contract Builder

将 CorpusSample 转换为 PlanningContract，用于 Writer 执行。
注意：生成的是剧情契约，而非测试契约。
"""

from typing import Optional
from src.writing.planning_contract import (
    PlanningContract,
    Intent,
    ContractMetadata,
    Execution,
    ExecutionUnit,
)
from experiments.phase12.corpus.models import CorpusSample


class ContractBuilder:
    """
    CorpusSample -> PlanningContract

    将测试样本转换为可执行剧情计划。
    不包含任何评估/测试语义。
    """

    def __init__(self, default_chapter: int = 1, default_scene_index: int = 0):
        self.default_chapter = default_chapter
        self.default_scene_index = default_scene_index

    def build(self, sample: CorpusSample) -> PlanningContract:
        """
        从样本生成 PlanningContract。
        """
        # 统一处理 category（支持 str 或 Enum）
        category = sample.category
        if hasattr(category, "value"):
            category = category.value

        # 剧情目标映射（非评估语义）
        goal_map = {
            "scene_transition": "完成自然的场景推进",
            "character_state": "展示角色状态变化",
            "dialogue_quality": "推进角色互动和信息交换",
            "planning_execution": "执行既定计划并产生结果",
            "runtime_state": "处理当前状态变化和异常",
        }

        conflict_map = {
            "scene_transition": "场景转换缺少因果连接",
            "character_state": "角色状态发生未知变化",
            "dialogue_quality": "角色交流需要推动剧情",
            "planning_execution": "计划执行存在阻碍",
            "runtime_state": "运行状态出现异常",
        }

        intent = Intent(
            goal=goal_map.get(category, "推进当前剧情"),
            conflict=conflict_map.get(category, "未知冲突"),
            expected_outcome="生成符合当前场景约束的后续剧情",
        )

        execution = Execution(
            units=[
                ExecutionUnit(
                    id="U1",
                    label="action",  # 改为 action
                    description=f"推进当前{category}相关剧情",
                    attributes={"corpus_sample_id": sample.id},
                )
            ]
        )

        return PlanningContract(
            scene_id=f"contract_{sample.id}",
            intent=intent,
            execution=execution,
            metadata=ContractMetadata(
                chapter=self.default_chapter,
                scene_index=self.default_scene_index,
            ),
        )