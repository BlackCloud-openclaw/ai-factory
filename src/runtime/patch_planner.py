# src/runtime/patch_planner.py
"""
Patch Planner - 根据诊断 IR 生成修复计划

职责：
1. 接收 FailureAnalysis（诊断 IR）
2. 根据失败类型生成 PatchPlan
3. 输出结构化的修复计划（不执行修复）

设计原则：
- 纯规划：只决定“改什么”，不执行
- IR 驱动：输入 FailureAnalysis，输出 PatchPlan
- 可扩展：每种失败类型可单独配置修复策略
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

from src.runtime.failure_analyzer import FailureAnalysis, FailureType, Severity, SuggestedStrategy


# ============================================================
# 修复动作
# ============================================================

class PatchActionType(Enum):
    """修复动作类型"""
    INSERT = "insert"          # 插入新内容
    REWRITE = "rewrite"        # 重写指定段落
    REPLACE = "replace"        # 替换整个层
    DELETE = "delete"          # 删除内容
    NONE = "none"              # 无需修复


@dataclass
class PatchAction:
    """单个修复动作"""
    action_type: PatchActionType
    target_layer: str          # "reasoning" | "justification" | "construction"
    instruction: str           # 具体的修复指令
    context: Optional[str] = None  # 需要修改的上下文（可选）
    priority: int = 1          # 优先级 1-5，1最高


# ============================================================
# 修复计划
# ============================================================

@dataclass
class PatchPlan:
    """修复计划"""
    actions: List[PatchAction]
    revision_required: bool
    estimated_risk: str        # "low" | "medium" | "high"
    estimated_cost: str        # "low" | "medium" | "high"
    reason: str = ""           # 为什么选择这个计划


# ============================================================
# Patch Planner 主类
# ============================================================

class PatchPlanner:
    """
    修复规划器 v1.0
    
    根据 FailureAnalysis 生成修复计划。
    版本：1.0
    """
    
    VERSION = "1.0"
    
    def plan(self, analysis: FailureAnalysis) -> PatchPlan:
        """
        根据单层诊断生成修复计划
        
        Args:
            analysis: FailureAnalysis 诊断 IR
            
        Returns:
            PatchPlan: 修复计划
        """
        failure_type = analysis.failure_type
        layer = analysis.layer
        
        # 根据失败类型分发
        if failure_type in [
            FailureType.NO_STATE,
            FailureType.JUSTIFICATION_NO_STATE,
            FailureType.CONSTRUCTION_NO_STATE,
        ]:
            return self._plan_insert(analysis)
        
        elif failure_type in [
            FailureType.STATE_MENTIONED_ONLY,
            FailureType.STATE_WRONG_LAYER,
            FailureType.STATE_NOT_BOUND_TO_REASONING,
            FailureType.REASONING_INSUFFICIENT,
            FailureType.JUSTIFICATION_MENTIONED_ONLY,
            FailureType.JUSTIFICATION_WEAK,
            FailureType.CONSTRUCTION_WEAK,
            FailureType.PREDICTION_UNCLEAR,
        ]:
            return self._plan_rewrite(analysis)
        
        elif failure_type in [
            FailureType.STATE_CONTRADICTS_POLICY,
            FailureType.PREDICTION_CHANGED,
        ]:
            return self._plan_reject(analysis)
        
        else:
            # 默认：保守的改写策略
            return self._plan_rewrite(analysis)
    
    def plan_all(self, analyses: List[FailureAnalysis]) -> List[PatchPlan]:
        """为多个诊断生成修复计划"""
        return [self.plan(a) for a in analyses]
    
    # ============================================================
    # 具体策略方法
    # ============================================================
    
    def _plan_insert(self, analysis: FailureAnalysis) -> PatchPlan:
        """插入策略 - 适用于 State 完全缺失的情况"""
        layer = analysis.layer
        
        instructions = {
            "reasoning": "在角色的推理过程中插入对 State 的显式引用。State 必须成为角色思考的一部分。",
            "justification": "在决策理由中插入 State 作为依据之一。",
            "construction": "在叙事实现中插入 State 的影响。",
        }
        
        instruction = instructions.get(layer, f"在 {layer} 层插入 State 相关内容。")
        
        return PatchPlan(
            actions=[
                PatchAction(
                    action_type=PatchActionType.INSERT,
                    target_layer=layer,
                    instruction=instruction,
                    priority=1,
                )
            ],
            revision_required=True,
            estimated_risk="low",
            estimated_cost="low",
            reason=f"State 完全缺失，插入新内容风险低，成本低",
        )
    
    def _plan_rewrite(self, analysis: FailureAnalysis) -> PatchPlan:
        """改写策略 - 适用于 State 存在但未充分使用的情况"""
        layer = analysis.layer
        
        instructions = {
            "reasoning": "重写角色的推理过程，确保 State 成为推理链条的一部分，而不是仅仅被提及。",
            "justification": "重写决策理由，确保 State 成为决策依据之一。",
            "construction": "重写叙事实现，确保 State 影响叙事方式。",
        }
        
        instruction = instructions.get(layer, f"重写 {layer} 层，确保 State 被充分使用。")
        
        return PatchPlan(
            actions=[
                PatchAction(
                    action_type=PatchActionType.REWRITE,
                    target_layer=layer,
                    instruction=instruction,
                    priority=2,
                )
            ],
            revision_required=True,
            estimated_risk="medium",
            estimated_cost="medium",
            reason=f"State 存在但未充分使用（{analysis.failure_type.value}），需改写相关段落",
        )
    
    def _plan_reject(self, analysis: FailureAnalysis) -> PatchPlan:
        """拒绝策略 - 适用于 State 违反 Policy 的情况"""
        layer = analysis.layer
        
        return PatchPlan(
            actions=[
                PatchAction(
                    action_type=PatchActionType.NONE,
                    target_layer=layer,
                    instruction=f"当前 Draft 违反了 Policy 要求（{analysis.failure_type.value}），建议完全重试。",
                    priority=5,
                )
            ],
            revision_required=False,  # 局部修复不可行，需要完全重试
            estimated_risk="high",
            estimated_cost="high",
            reason=f"Policy 被违反（{analysis.failure_type.value}），局部修复风险高，建议重新生成",
        )


# ============================================================
# 便捷函数
# ============================================================

def plan_patch(analysis: FailureAnalysis) -> PatchPlan:
    """便捷函数：为单层诊断生成修复计划"""
    planner = PatchPlanner()
    return planner.plan(analysis)