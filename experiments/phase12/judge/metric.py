from typing import Optional, Dict, Any, cast

from ..model import EvaluationContext, MetricResult, MetricState
from ..metrics.protocol import Metric
from ..metrics.mixins import AverageAggregateMixin
from .client import LLMJudgeClient
from .models import JudgeDimension
from .context import JudgeContext  # 这里导入是安全的，因为 model.py 不再导入 judge


class BaseLLMJudgeMetric(AverageAggregateMixin, Metric):
    """LLM Judge 基类，所有 Judge Metric 共享同一 Client。"""

    name: str
    version: str = "1.0"
    # weight 已移除，由 config 统一管理

    def __init__(
        self,
        dimension: JudgeDimension,
        client: LLMJudgeClient,
        pass_threshold: float = 0.5,
    ):
        self._dimension = dimension
        self._client = client
        self._pass_threshold = pass_threshold

    async def evaluate(self, ctx: EvaluationContext) -> MetricResult:
        try:
            result = await self._client.evaluate(
                dimension=self._dimension,
                text=ctx.scene_text,
                context=self._build_context(ctx),
                use_cache=True,
            )

            score = result.score
            passed = score >= self._pass_threshold

            return MetricResult(
                name=self.name,
                score=score,
                state=MetricState.OK,
                raw_value=result.score,
                details={
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                    "tokens_used": result.tokens_used,
                    "elapsed_ms": result.elapsed_ms,
                    "dimension": self._dimension.value,
                },
                passed=passed,
            )
        except Exception as e:
            return MetricResult(
                name=self.name,
                score=None,
                state=MetricState.ERROR,
                details={"error": str(e)},
                passed=False,
            )

    def _build_context(self, ctx: EvaluationContext) -> Dict[str, Any]:
        judge_ctx = ctx.judge_context
        if judge_ctx is None:
            return {}
        # 如果是 JudgeContext 实例，访问属性
        if isinstance(judge_ctx, JudgeContext):
            return {
                "previous_scene_text": judge_ctx.previous_scene_text,
                "character_summary": judge_ctx.character_summary,
                # 其他字段...
            }
        return {}


class ContinuityJudgeMetric(BaseLLMJudgeMetric):
    name = "continuity"

    def __init__(self, client: LLMJudgeClient, pass_threshold: float = 0.5):
        super().__init__(dimension=JudgeDimension.CONTINUITY, client=client, pass_threshold=pass_threshold)

    def _build_context(self, ctx: EvaluationContext) -> Dict[str, Any]:
        judge_ctx = ctx.judge_context or JudgeContext()
        return {"scene_before": judge_ctx.previous_scene_text or "（无上一场景）"}


class CharacterJudgeMetric(BaseLLMJudgeMetric):
    name = "character"

    def __init__(self, client: LLMJudgeClient, pass_threshold: float = 0.5):
        super().__init__(dimension=JudgeDimension.CHARACTER, client=client, pass_threshold=pass_threshold)

    def _build_context(self, ctx: EvaluationContext) -> Dict[str, Any]:
        judge_ctx = ctx.judge_context or JudgeContext()
        return {"character_info": judge_ctx.character_summary or "（无角色信息）"}


class DialogueJudgeMetric(BaseLLMJudgeMetric):
    name = "dialogue"

    def __init__(self, client: LLMJudgeClient, pass_threshold: float = 0.5):
        super().__init__(dimension=JudgeDimension.DIALOGUE, client=client, pass_threshold=pass_threshold)


class FlowJudgeMetric(BaseLLMJudgeMetric):
    name = "flow"

    def __init__(self, client: LLMJudgeClient, pass_threshold: float = 0.5):
        super().__init__(dimension=JudgeDimension.FLOW, client=client, pass_threshold=pass_threshold)