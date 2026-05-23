# src/orchestrator/phase_resolver.py
from src.orchestrator.state import AgentState
from src.orchestrator.state_patch import WorkflowPhase


class WorkflowPhaseResolver:
    """处理 phase 的推断与向后兼容（resume 时非常重要）"""
    
    @staticmethod
    def resolve(state: AgentState) -> WorkflowPhase:
        if state.phase is not None:
            return state.phase
        
        # 向后兼容推断（resume 旧 checkpoint）
        if state.scene_plan_list and not state.scene_text:
            return WorkflowPhase.WRITING
        if state.validation_result:
            return WorkflowPhase.VALIDATING
        if state.current_chapter and not state.scene_plan_list:
            return WorkflowPhase.PLANNING
        return WorkflowPhase.PLANNING