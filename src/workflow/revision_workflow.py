# src/workflow/revision_workflow.py

"""
Revision Workflow — 编排 Runtime Compiler 和 LLM 调用，执行修订闭环。

Phase 8 适配：
- 使用 build_default_snapshot() 从 default_runtime 获取 Snapshot
- 导入路径从 builder 改为 default_runtime
- 支持外部注入 snapshot，支持缓存控制
"""

import time
import logging
from typing import Dict, Any, Optional, Callable, Awaitable, Tuple

from src.runtime.observation_compiler import ObservationCompiler
from src.runtime.validator import Validator
from src.runtime.edit_compiler import EditCompiler
from src.runtime.patch_renderer import PatchRenderer
from src.runtime import RuntimeSnapshot, RuntimeConfig
from src.runtime.default_runtime import build_default_snapshot

logger = logging.getLogger("workflow.revision")


class RevisionWorkflow:
    """
    修订工作流 - Phase 6.5 / Phase 7 兼容版 / Phase 8 适配版

    职责：编排 Runtime Compiler 和 LLM 调用，执行修订闭环。
    """

    def __init__(
        self,
        llm_executor: Optional[Callable[[str], Awaitable[str]]] = None,
        layer_targets: Optional[Dict[str, str]] = None,
        max_rounds: int = 2,
        compliance_threshold: float = 0.7,
        enable_revision: bool = False,
        snapshot: Optional[RuntimeSnapshot] = None,
        surface_ids: Optional[Tuple[str, ...]] = None,
        cache_snapshot: bool = True,
    ):
        """
        Args:
            llm_executor: 异步函数，接收 prompt 返回修订后文本
            layer_targets: 各层的目标等级
            max_rounds: 最大修订轮数
            compliance_threshold: 合规阈值，高于此值跳过修订
            enable_revision: 是否启用修订
            snapshot: 外部注入的 RuntimeSnapshot（优先）
            surface_ids: 备用构建时启用的 Surface ID
            cache_snapshot: 是否缓存构建的 Snapshot
        """
        self.llm_executor = llm_executor
        self.layer_targets = layer_targets or {
            "reasoning": "enhanced",
            "justification": "enhanced",
            "construction": "enhanced",
            "prediction": "enhanced",
        }
        self.max_rounds = max_rounds
        self.compliance_threshold = compliance_threshold
        self.enable_revision = enable_revision

        self._obs_compiler = ObservationCompiler()
        self._validator = Validator()
        self._edit_compiler = EditCompiler()
        self._patch_renderer = PatchRenderer()

        # Phase 7/8: RuntimeSnapshot
        self._snapshot = snapshot
        self._surface_ids = surface_ids or ("reasoning",)
        self._cache_snapshot = cache_snapshot
        self._cached_snapshot: Optional[RuntimeSnapshot] = None
        self._snapshot_built = False

    def _get_snapshot(self) -> RuntimeSnapshot:
        """
        获取 RuntimeSnapshot。

        优先级：
        1. 外部注入的 snapshot
        2. 缓存的 snapshot（如果启用）
        3. 通过 default_runtime 构建
        """
        if self._snapshot is not None:
            return self._snapshot

        if self._cache_snapshot and self._cached_snapshot is not None:
            return self._cached_snapshot

        # 通过 Composition Root 构建默认 Snapshot
        config = RuntimeConfig(
            enabled_surfaces=self._surface_ids,
            # diagnostics 默认 False，由调用方按需开启
        )
        snapshot = build_default_snapshot(config=config)

        if self._cache_snapshot:
            self._cached_snapshot = snapshot

        self._snapshot_built = True
        logger.info(
            "RuntimeSnapshot built: surfaces=%s",
            snapshot.get_surface_ids(),
        )
        return snapshot

    async def execute(self, draft: str) -> Dict[str, Any]:
        """
        执行修订闭环，返回 ExecutionResult。

        Args:
            draft: 待修订的草稿文本

        Returns:
            Dict 包含：
            - final_text: 最终文本
            - compliance: 最终合规率
            - before_compliance: 修订前合规率
            - after_compliance: 修订后合规率
            - compliance_delta: 合规率变化
            - stages: 各阶段执行详情
            - artifacts: 各阶段产物
        """
        current_draft = draft
        stages = []
        artifacts: Dict[str, Any] = {}

        snapshot = self._get_snapshot()

        # 仅记录 warning，不 assert（Workflow 不承担配置验证职责）
        if "reasoning" not in snapshot.get_surface_ids():
            logger.warning(
                "Snapshot missing 'reasoning' surface. Available: %s",
                snapshot.get_surface_ids(),
            )

        # ---- Stage 1: Validation ----
        start_time = time.perf_counter()
        ir = self._obs_compiler.compile(current_draft, snapshot)
        report = self._validator.validate(snapshot, ir)
        compliance = report.overall_compliance
        duration_ms = (time.perf_counter() - start_time) * 1000

        layers = []
        for layer_result in report.layer_results:
            for evidence in layer_result.evidence_list:
                layers.append({
                    "layer": layer_result.layer,
                    "observed": list(evidence.present_patterns),
                    "missing": list(evidence.missing_pattern_types),
                })
            if not layer_result.evidence_list and layer_result.compliant:
                layers.append({
                    "layer": layer_result.layer,
                    "observed": [],
                    "missing": [],
                })

        stages.append({
            "stage": "validation",
            "status": "completed",
            "duration_ms": duration_ms,
            "payload": {
                "compliance": compliance,
                "layers": layers,
            },
        })
        artifacts["validation"] = report

        # 如果合规或未启用修订，直接返回
        if compliance >= self.compliance_threshold or not self.enable_revision:
            return {
                "final_text": current_draft,
                "compliance": compliance,
                "before_compliance": compliance,
                "after_compliance": compliance,
                "compliance_delta": 0.0,
                "stages": stages,
                "artifacts": artifacts,
            }

        # ---- Stage 2: Edit Planning ----
        start_time = time.perf_counter()
        plan = self._edit_compiler.compile_with_snapshot(
            snapshot,
            report,
            current_draft,
            ir,
            diagnosis_id="REV_001",
        )
        action_count = len(plan.actions) if plan else 0
        duration_ms = (time.perf_counter() - start_time) * 1000

        if action_count == 0:
            stages.append({
                "stage": "edit_plan",
                "status": "skipped",
                "duration_ms": duration_ms,
                "payload": {"action_count": 0, "reason": "no_actions"},
            })
            return {
                "final_text": current_draft,
                "compliance": compliance,
                "before_compliance": compliance,
                "after_compliance": compliance,
                "compliance_delta": 0.0,
                "stages": stages,
                "artifacts": artifacts,
            }

        stages.append({
            "stage": "edit_plan",
            "status": "completed",
            "duration_ms": duration_ms,
            "payload": {"action_count": action_count},
        })
        artifacts["edit_plan"] = plan

        # ---- Stage 3: Patch Rendering ----
        start_time = time.perf_counter()
        rendered = self._patch_renderer.render(plan, ir)
        duration_ms = (time.perf_counter() - start_time) * 1000

        stages.append({
            "stage": "patch_render",
            "status": "completed",
            "duration_ms": duration_ms,
            "payload": {"prompt_length": len(rendered.full_prompt)},
        })
        artifacts["rendered_prompt"] = rendered

        # ---- Stage 4: LLM Execution ----
        start_time = time.perf_counter()
        llm_success = False
        finish_reason = "unknown"
        llm_output = None
        error = None

        if self.llm_executor:
            try:
                llm_output = await self.llm_executor(rendered.full_prompt)
                llm_success = True
                finish_reason = "stop"
                current_draft = llm_output
            except Exception as e:
                error = str(e)
                finish_reason = (
                    "length"
                    if "truncated" in error.lower() or "missing closing" in error.lower()
                    else "error"
                )
        else:
            error = "llm_executor is None"
            finish_reason = "error"

        duration_ms = (time.perf_counter() - start_time) * 1000

        stages.append({
            "stage": "llm",
            "status": "completed" if llm_success else "failed",
            "duration_ms": duration_ms,
            "payload": {
                "success": llm_success,
                "finish_reason": finish_reason,
                "error": error,
            },
        })
        if llm_output:
            artifacts["llm_output"] = llm_output

        if not llm_success:
            return {
                "final_text": current_draft,
                "compliance": compliance,
                "before_compliance": compliance,
                "after_compliance": compliance,
                "compliance_delta": 0.0,
                "stages": stages,
                "artifacts": artifacts,
            }

        # ---- Stage 5: Revalidation ----
        start_time = time.perf_counter()
        final_ir = self._obs_compiler.compile(current_draft, snapshot)
        final_report = self._validator.validate(snapshot, final_ir)
        final_compliance = final_report.overall_compliance
        duration_ms = (time.perf_counter() - start_time) * 1000

        stages.append({
            "stage": "revalidation",
            "status": "completed",
            "duration_ms": duration_ms,
            "payload": {"final_compliance": final_compliance},
        })
        artifacts["revalidation"] = final_report

        return {
            "final_text": current_draft,
            "compliance": final_compliance,
            "before_compliance": compliance,
            "after_compliance": final_compliance,
            "compliance_delta": final_compliance - compliance,
            "stages": stages,
            "artifacts": artifacts,
        }