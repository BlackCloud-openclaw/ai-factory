# src/writing/audit/coordinator.py
"""
Phase 10.3.1: AuditCoordinator — 无状态审计管道编排器
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Any, Sequence, TYPE_CHECKING
from uuid import UUID, uuid4

from .trace import ExecutionTrace
from .collector import TraceCollector
from .preservation import PreservationAnalyzer
from .attribution import AttributionAnalyzer
from .budget import MetricBudgetAnalyzer
from .priority import PriorityEngine
from .reporter import Reporter, ComprehensiveReport
from .payload_resolver import PayloadResolver

if TYPE_CHECKING:
    from .trace import ExecutionTrace

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuditConfig:
    """审计配置。"""
    enabled: bool = True
    enable_preservation: bool = True
    enable_attribution: bool = True
    enable_budget: bool = True
    enable_priority: bool = True
    fields: Sequence[str] = field(default_factory=lambda: [
        "goal", "conflict", "outcome", "must_events", "characters", "constraints", "scene_spec"
    ])
    budget_metric: str = "tokens"
    auto_report: bool = True


class AuditCoordinator:
    """
    审计管道编排器（无状态）。
    """

    def __init__(
        self,
        resolver: Optional[PayloadResolver] = None,
        config: Optional[AuditConfig] = None,
    ):
        self._resolver = resolver
        self._config = config or AuditConfig()

    def start(
        self,
        novel_id: str,
        volume: int,
        chapter: int,
        scene_idx: int,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "AuditContext":
        execution_id = uuid4()
        collector = None
        if self._config.enabled:
            collector = TraceCollector(
                novel_id=novel_id,
                volume=volume,
                chapter=chapter,
                scene_idx=scene_idx,
                metadata=metadata,
                execution_id=execution_id,
            )
        return AuditContext(
            coordinator=self,
            collector=collector,
            execution_id=execution_id,
        )

    def audit(self, novel_id: str, volume: int, chapter: int, scene_idx: int, **kwargs):
        return self.start(novel_id, volume, chapter, scene_idx, **kwargs)

    def _generate_report(self, trace: ExecutionTrace) -> Optional[ComprehensiveReport]:
        """从 Trace 生成报告（异常隔离：捕获所有异常，不向上传播）。"""
        if not self._config.enabled:
            return None

        try:
            # 1. Preservation
            pres_report = None
            if self._config.enable_preservation:
                analyzer = PreservationAnalyzer(
                    resolver=self._resolver,
                    fields=self._config.fields,
                )
                pres_report = analyzer.analyze(trace)

            # 2. Attribution
            attr_report = None
            if self._config.enable_attribution and pres_report is not None:
                analyzer = AttributionAnalyzer(
                    resolver=self._resolver,
                )
                attr_report = analyzer.analyze(trace, pres_report)

            # 3. Budget
            budget_report = None
            if self._config.enable_budget:
                analyzer = MetricBudgetAnalyzer(metric=self._config.budget_metric)
                budget_report = analyzer.analyze(trace)

            # 4. Priority
            priority_report = None
            if self._config.enable_priority and all([pres_report, attr_report, budget_report]):
                engine = PriorityEngine()
                priority_report = engine.analyze(
                    execution_id=str(trace.execution_id),
                    preservation_report=pres_report,
                    attribution_report=attr_report,
                    budget_report=budget_report,
                )

            # 5. Reporter
            if self._config.auto_report and all([pres_report, attr_report, budget_report, priority_report]):
                reporter = Reporter()
                return reporter.generate(
                    preservation_report=pres_report,
                    attribution_report=attr_report,
                    budget_report=budget_report,
                    priority_report=priority_report,
                )

            return None

        except Exception as e:
            # 审计失败不影响 Writer 执行，仅记录日志
            logger.error(f"Audit report generation failed: {e}", exc_info=True)
            return None


class AuditContext:
    def __init__(
        self,
        coordinator: AuditCoordinator,
        collector: Optional[TraceCollector],
        execution_id: UUID,
    ):
        self._coordinator = coordinator
        self._collector = collector
        self._execution_id = execution_id
        self._report: Optional[ComprehensiveReport] = None
        self._trace: Optional[ExecutionTrace] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._collector is not None:
            self._trace = self._collector.finish()
            self._report = self._coordinator._generate_report(self._trace)

    @property
    def execution_id(self) -> UUID:
        return self._execution_id

    @property
    def trace(self) -> Optional[ExecutionTrace]:
        return self._trace

    @property
    def report(self) -> Optional[ComprehensiveReport]:
        return self._report

    @property
    def collector(self) -> Optional[TraceCollector]:
        return self._collector

    def record_stage(self, stage: str, inputs: Optional[dict] = None, outputs: Optional[dict] = None, **kwargs):
        if self._collector is not None:
            self._collector.record_stage(stage, inputs, outputs, kwargs)