# src/runtime/patch_compiler.py
"""
Patch Compiler - RevisionDecision → PatchPlan (IR)

职责：将 RevisionDecision 编译为纯 IR 的 PatchPlan。
不生成自然语言，只输出结构化操作。
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

from src.runtime.failure_analyzer import FailureDiagnosis
from src.runtime.revision_controller import RevisionDecision, RevisionStrategy, RevisionRisk


class PatchOperation(Enum):
    INSERT = "insert"
    REWRITE = "rewrite"
    REPLACE = "replace"


@dataclass(frozen=True)
class PatchAction:
    layer: str
    operation: PatchOperation


@dataclass(frozen=True)
class PatchPlan:
    actions: List[PatchAction]
    revision_required: bool
    estimated_risk: RevisionRisk
    estimated_cost: str
    reason: str


class PatchCompiler:
    """
    Patch Compiler v1.0 - 将 RevisionDecision 编译为 PatchPlan (IR)
    """
    
    VERSION = "1.0"
    
    def compile(self, decision: RevisionDecision, diagnosis: FailureDiagnosis) -> PatchPlan:
        if not decision.should_revise or decision.strategy == RevisionStrategy.SKIP:
            return PatchPlan(
                actions=[],
                revision_required=False,
                estimated_risk=RevisionRisk.LOW,
                estimated_cost="low",
                reason="Controller 决定跳过修订",
            )
        
        if decision.strategy == RevisionStrategy.FULL_RETRY:
            return PatchPlan(
                actions=[],
                revision_required=True,
                estimated_risk=RevisionRisk.CRITICAL,
                estimated_cost="high",
                reason=decision.rationale,
            )
        
        # 根据策略和目标层生成动作
        actions = []
        for layer in decision.target_layers:
            op = self._operation_for_strategy(decision.strategy)
            actions.append(PatchAction(layer=layer, operation=op))
        
        return PatchPlan(
            actions=actions,
            revision_required=len(actions) > 0,
            estimated_risk=decision.estimated_risk,
            estimated_cost="medium",
            reason=decision.rationale,
        )
    
    def _operation_for_strategy(self, strategy: RevisionStrategy) -> PatchOperation:
        if strategy == RevisionStrategy.INSERT:
            return PatchOperation.INSERT
        elif strategy == RevisionStrategy.REWRITE:
            return PatchOperation.REWRITE
        elif strategy == RevisionStrategy.REPLACE:
            return PatchOperation.REPLACE
        else:
            return PatchOperation.REWRITE