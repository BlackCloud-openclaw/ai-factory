"""
Phase 12.2B-3: Corpus Regenerator

将 v1.1 CorpusSample 转换为 v2.0 YAML。
职责：单样本执行，不包含批量逻辑。
"""

from pathlib import Path
from typing import Optional, Protocol, runtime_checkable
from dataclasses import dataclass

from src.writing.planning_contract import PlanningContract
from src.writing.evaluation import EvaluationSnapshot, RuntimeMetrics
from experiments.phase12.corpus.models import CorpusSample
from experiments.phase12.corpus.contract_builder import ContractBuilder
from experiments.phase12.corpus.exporter import CorpusExporter


@runtime_checkable
class WriterProtocol(Protocol):
    """Writer 的最小协议，用于解耦真实 Writer"""
    async def execute_with_snapshot(
        self,
        contract: PlanningContract,
        scene_before: str,
        previous_scene_text: Optional[str] = None,
        character_summary: Optional[dict] = None,
        world_summary: Optional[dict] = None,
        novel_id: str = "",
        volume: int = 1,
        chapter: int = 1,
        scene_idx: int = 0,
    ) -> EvaluationSnapshot:
        ...


@dataclass(frozen=True)
class RegenerateResult:
    """单样本再生结果"""
    sample_id: str
    category: str
    output_path: Path
    snapshot: EvaluationSnapshot
    success: bool
    error: Optional[str] = None


class CorpusRegenerator:
    """
    Corpus 再生器：单样本执行器。

    职责：
    1. 从 CorpusSample 构建 PlanningContract
    2. 调用 Writer 生成 EvaluationSnapshot
    3. 使用 Exporter 导出 v2.0 YAML
    """

    def __init__(
        self,
        writer: WriterProtocol,
        output_dir: Path,
        version: str = "2.0",
    ):
        self.writer = writer
        self.output_dir = Path(output_dir)
        self.version = version
        self._builder = ContractBuilder()
        self._exporter = CorpusExporter(
            output_dir=output_dir,
            version=version,
        )

    async def regenerate_sample(
        self,
        sample: CorpusSample,
        category: Optional[str] = None,
        previous_scene_text: Optional[str] = None,
        character_summary: Optional[dict] = None,
        world_summary: Optional[dict] = None,
        novel_id: str = "",
        chapter: int = 1,
        scene_idx: int = 0,
    ) -> RegenerateResult:
        """
        重新生成单个样本。

        Args:
            sample: v1.1 CorpusSample
            category: 输出类别（默认使用 sample.category）
            previous_scene_text: 上一场景文本（用于 JudgeContext）
            character_summary: 角色摘要
            world_summary: 世界摘要
            novel_id: 小说 ID
            chapter: 章节号
            scene_idx: 场景序号

        Returns:
            RegenerateResult
        """
        sample_id = sample.id
        category = category or sample.category
        if hasattr(category, "value"):
            category = category.value

        try:
            # 1. 构建 Contract
            contract = self._builder.build(sample)

            # 2. 调用 Writer
            snapshot = await self.writer.execute_with_snapshot(
                contract=contract,
                scene_before=sample.scene_before,
                previous_scene_text=previous_scene_text,
                character_summary=character_summary,
                world_summary=world_summary,
                
                
                
            )

            # 3. 导出 YAML
            output_path = self._exporter.export(
                snapshot=snapshot,
                category=category,
                failure_modes=[category],
                sample_id=f"corpus.{category}.regenerated.{sample_id}",
            )

            return RegenerateResult(
                sample_id=sample_id,
                category=category,
                output_path=output_path,
                snapshot=snapshot,
                success=True,
            )

        except Exception as e:
            return RegenerateResult(
                sample_id=sample_id,
                category=category,
                output_path=Path(),
                snapshot=EvaluationSnapshot(
                    scene_before="",
                    scene_after="",
                    runtime_metrics=RuntimeMetrics(
                        error_count=1,
                    ),
                ),
                success=False,
                error=str(e),
            )