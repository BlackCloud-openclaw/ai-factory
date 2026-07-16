# src/workflow/revision_workflow.py

import time
from typing import Dict, Any, Optional, List, Callable, Awaitable
import logging

from src.runtime.observation_compiler import ObservationCompiler
from src.runtime.validator import Validator
from src.runtime.edit_compiler import EditCompiler
from src.runtime.patch_renderer import PatchRenderer

logger = logging.getLogger("workflow.revision")


class RevisionWorkflow:
    """
    修订工作流 - Phase 6.5
    职责：编排 Runtime Compiler 和 LLM 调用，执行修订闭环。
    返回 ExecutionResult（内部对象，非公共 API）。
    """

    def __init__(
        self,
        llm_executor: Optional[Callable[[str], Awaitable[str]]] = None,
        layer_targets: Optional[Dict[str, str]] = None,
        max_rounds: int = 2,
        compliance_threshold: float = 0.7,
        enable_revision: bool = False,
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

    async def execute(self, draft: str) -> Dict[str, Any]:
        """
        执行修订闭环，返回 ExecutionResult（内部对象，非公共 API）
        """
        current_draft = draft
        stages = []
        artifacts: Dict[str, Any] = {}

        # ---- Stage 1: Validation ----
        start_time = time.perf_counter()
        ir = self._obs_compiler.compile(current_draft)
        report = self._validator.validate(ir, self.layer_targets)
        compliance = report.overall_compliance
        duration_ms = (time.perf_counter() - start_time) * 1000

        # ---- Phase 6.5.1: 映射 Layer 详情（只做提取，不做推导） ----
        layers = []
        for layer_result in report.layer_results:
            # 每个 layer_result 可能包含多个 evidence（不同句子）
            for evidence in layer_result.evidence_list:
                layers.append({
                    "layer": layer_result.layer,
                    "observed": list(evidence.present_patterns),
                    "missing": list(evidence.missing_pattern_types),
                    # anchor_sentence_id 暂不输出，但已准备好
                })
            # 如果该层没有 evidence，但合规，添加一个空记录
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

        # 如果合规或未启用修订，直接返回
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
        plan = self._edit_compiler.compile(ir, report, diagnosis_id="REV_001")
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
        final_ir = self._obs_compiler.compile(current_draft)
        final_report = self._validator.validate(final_ir, self.layer_targets)
        final_compliance = final_report.overall_compliance
        duration_ms = (time.perf_counter() - start_time) * 1000

        stages.append({
            "stage": "revalidation",
            "status": "completed",
            "duration_ms": duration_ms,
            "payload": {"final_compliance": final_compliance}
        })
        artifacts["revalidation"] = final_report

        # 返回最终结果
        return {
            "final_text": current_draft,
            "compliance": final_compliance,
            "before_compliance": compliance,
            "after_compliance": final_compliance,
            "compliance_delta": final_compliance - compliance,
            "stages": stages,
            "artifacts": artifacts
        }