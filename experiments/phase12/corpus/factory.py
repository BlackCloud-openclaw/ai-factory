"""
ContextFactory：将 CorpusSample 转换为 EvaluationContext
"""

from typing import Optional, List, Any

from .models import CorpusSample
from .adapter import RuntimeArtifactAdapter, PlanningContractDeserializer, RuntimeSnapshotDeserializer, NarrativeEventDeserializer
from .judge_context_factory import JudgeContextFactory
from ..model import EvaluationContext
from src.writing.evaluation import EvaluationSnapshot


class ContextFactory:
    """从 CorpusSample 创建 EvaluationContext"""

    def __init__(
        self,
        adapter: Optional[RuntimeArtifactAdapter] = None,
        judge_factory: Optional[JudgeContextFactory] = None,
    ):
        self._adapter = adapter or RuntimeArtifactAdapter()
        self._judge_factory = judge_factory or JudgeContextFactory()

    def create(self, sample: CorpusSample) -> EvaluationContext:
        """将单个样本转换为 EvaluationContext"""
        # 1. 使用 Adapter 转换 Runtime 对象（从通用 Mapping 转换）
        planning = self._adapter.to_planning_contract(sample.artifacts.planning_contract)
        before = self._adapter.to_snapshot(sample.artifacts.snapshot_before)
        after = self._adapter.to_snapshot(sample.artifacts.snapshot_after)
        events = self._adapter.to_events(sample.artifacts.events)

        # 2. 使用 JudgeContextFactory 构建 JudgeContext
        judge_context = self._judge_factory.create(sample)

        # 3. 组装 EvaluationContext
        return EvaluationContext(
            planning_contract=planning,
            scene_text=sample.scene_after,
            events=events,
            snapshot_before=before,
            snapshot_after=after,
            runtime_metrics=sample.artifacts.runtime_metrics,
            judge_context=judge_context,
            novel_id=sample.id,
            volume=0,
            chapter=0,
            scene_idx=0,
        )

    def create_batch(self, samples: List[CorpusSample]) -> List[EvaluationContext]:
        return [self.create(s) for s in samples]
    
    @staticmethod
    def from_snapshot(snapshot: EvaluationSnapshot) -> 'EvaluationContext':
        from experiments.phase12.model import EvaluationContext

        return EvaluationContext(
            scene_text=snapshot.scene_after,
            events=snapshot.artifacts.get("events", []),
            snapshot_before=None,
            snapshot_after=None,
            runtime_metrics=snapshot.runtime_metrics,
            revision_result=snapshot.revision_result,
            judge_context=snapshot.judge_context,
            # 可选字段设默认
            planning_contract=None,
            novel_id="",
            volume=0,
            chapter=0,
            scene_idx=0,
            sample_id="",
        )