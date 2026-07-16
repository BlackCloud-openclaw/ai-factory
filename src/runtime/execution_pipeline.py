# src/runtime/execution_pipeline.py
"""
执行层完整流水线：Draft → Validator → Analyzer → Planner → Revision → Patched Draft
"""

from typing import Optional, List
import logging

from src.runtime.validator import validate_draft
from src.runtime.failure_analyzer import analyze_failures
from src.runtime.patch_planner import PatchPlanner, PatchPlan, PatchAction, PatchActionType
from src.runtime.revision_engine import RevisionEngine, RevisionResult
from src.runtime.compiler import LayerControlTargets

logger = logging.getLogger(__name__)


async def execute_with_diagnosis(
    draft: str,
    targets: LayerControlTargets,
    llm_api_base: str,
    llm_model: str,
    enable_revision: bool = True,
) -> dict:
    """
    执行完整的诊断 → 修复流水线
    
    Args:
        draft: Writer 生成的初稿
        targets: LayerControlTargets (IR)
        llm_api_base: LLM API 地址
        llm_model: LLM 模型名称
        enable_revision: 是否启用修复
        
    Returns:
        dict: 包含合规报告、诊断、修复计划、修复结果
    """
    # 1. Validator
    report = validate_draft(draft, targets)
    
    result = {
        "compliance_report": report,
        "diagnosis": None,
        "patch_plan": None,
        "revision_result": None,
        "final_text": draft,
    }
    
    # 2. 如果合规，直接返回
    if not report.revision_required:
        result["final_text"] = draft
        return result
    
    # 3. FailureAnalyzer
    diagnosis = analyze_failures(report, draft, targets)
    result["diagnosis"] = diagnosis
    
    # 4. 如果不需要修复或禁用修复，返回
    if not enable_revision or not diagnosis.requires_attention:
        return result
    
    # 5. 为每个失败生成 PatchPlan
    planner = PatchPlanner()
    plans = []
    for analysis in diagnosis.analyses:
        plan = planner.plan(analysis)
        plans.append(plan)
    
    # 6. 合并所有 PatchPlan 为一个
    merged_plan = _merge_plans(plans)
    result["patch_plan"] = merged_plan
    
    # 7. 执行修订
    engine = RevisionEngine(llm_api_base, llm_model)
    revision_result = await engine.revise(draft, merged_plan)
    result["revision_result"] = revision_result
    result["final_text"] = revision_result.patched_text
    
    return result


def _merge_plans(plans: List[PatchPlan]) -> PatchPlan:
    """合并多个 PatchPlan 为一个"""
    all_actions = []
    for plan in plans:
        all_actions.extend(plan.actions)
    
    # 去重：同一 layer 只保留一个动作
    seen_layers = set()
    unique_actions = []
    for action in all_actions:
        if action.target_layer not in seen_layers:
            seen_layers.add(action.target_layer)
            unique_actions.append(action)
    
    # 如果没有动作，返回一个空计划
    if not unique_actions:
        return PatchPlan(
            actions=[],
            revision_required=False,
            estimated_risk="low",
            estimated_cost="low",
            reason="无需修复",
        )
    
    return PatchPlan(
        actions=unique_actions,
        revision_required=True,
        estimated_risk="medium",
        estimated_cost="medium",
        reason=f"合并 {len(plans)} 个修复计划",
    )