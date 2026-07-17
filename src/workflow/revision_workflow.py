# src/workflow/revision_workflow.py

import time
import logging
from typing import Dict, Any, Optional, Callable, Awaitable, Tuple

from src.runtime.observation_compiler import ObservationCompiler
from src.runtime.validator import Validator
from src.runtime.edit_compiler import EditCompiler
from src.runtime.patch_renderer import PatchRenderer
from src.runtime import RuntimeSnapshot
from src.runtime.builder import build_default_snapshot  # ✅ 使用 builder 中的 helper

logger = logging.getLogger("workflow.revision")


class RevisionWorkflow:
    def __init__(
        self,
        llm_executor: Optional[Callable[[str], Awaitable[str]]] = None,
        layer_targets: Optional[Dict[str, str]] = None,
        max_rounds: int = 2,
        compliance_threshold: float = 0.7,
        enable_revision: bool = False,
        snapshot: Optional[RuntimeSnapshot] = None,
        surface_ids: Optional[Tuple[str, ...]] = None,
        cache_snapshot: bool = True,  # ✅ 显式控制缓存
    ):
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

        # Phase 7: RuntimeSnapshot
        self._snapshot = snapshot
        self._surface_ids = surface_ids or ("reasoning",)
        self._cache_snapshot = cache_snapshot
        self._cached_snapshot: Optional[RuntimeSnapshot] = None
        self._snapshot_built = False

    def _get_snapshot(self) -> RuntimeSnapshot:
        """获取 RuntimeSnapshot：优先注入，否则延迟构建（支持缓存）"""
        if self._snapshot is not None:
            return self._snapshot

        if self._cache_snapshot and self._cached_snapshot is not None:
            return self._cached_snapshot

        # ✅ 使用 builder 中的统一 helper
        snapshot = build_default_snapshot(self._surface_ids)

        if self._cache_snapshot:
            self._cached_snapshot = snapshot

        self._snapshot_built = True
        logger.info(
            f"RuntimeSnapshot built: surfaces={snapshot.get_surface_ids()}"
        )
        return snapshot

    async def execute(self, draft: str) -> Dict[str, Any]:
        current_draft = draft
        stages = []
        artifacts: Dict[str, Any] = {}

        snapshot = self._get_snapshot()

        # 仅记录 warning，不 assert
        if "reasoning" not in snapshot.get_surface_ids():
            logger.warning(
                f"Snapshot missing 'reasoning' surface. Available: {snapshot.get_surface_ids()}"
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
                "layers": layers
            }
        })
        artifacts["validation"] = report

        if compliance >= self.compliance_threshold or not self.enable_revision:
            return {
                "final_text": current_draft,
                "compliance": compliance,
                "before_compliance": compliance,
                "after_compliance": compliance,
                "compliance_delta": 0.0,
                "stages": stages,
                "artifacts": artifacts
            }

        # ---- Stage 2: Edit Planning ----
        start_time = time.perf_counter()
        plan = self._edit_compiler.compile_with_snapshot(
            snapshot, report, current_draft, ir, diagnosis_id="REV_001"
        )
        action_count = len(plan.actions) if plan else 0
        duration_ms = (time.perf_counter() - start_time) * 1000

        if action_count == 0:
            stages.append({
                "stage": "edit_plan",
                "status": "skipped",
                "duration_ms": duration_ms,
                "payload": {"action_count": 0, "reason": "no_actions"}
            })
            return {
                "final_text": current_draft,
                "compliance": compliance,
                "before_compliance": compliance,
                "after_compliance": compliance,
                "compliance_delta": 0.0,
                "stages": stages,
                "artifacts": artifacts
            }

        stages.append({
            "stage": "edit_plan",
            "status": "completed",
            "duration_ms": duration_ms,
            "payload": {"action_count": action_count}
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
            "payload": {"prompt_length": len(rendered.full_prompt)}
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
                finish_reason = "length" if ("truncated" in error.lower() or "missing closing" in error.lower()) else "error"
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
                "error": error
            }
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
                "artifacts": artifacts
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
            "payload": {"final_compliance": final_compliance}
        })
        artifacts["revalidation"] = final_report

        return {
            "final_text": current_draft,
            "compliance": final_compliance,
            "before_compliance": compliance,
            "after_compliance": final_compliance,
            "compliance_delta": final_compliance - compliance,
            "stages": stages,
            "artifacts": artifacts
        }