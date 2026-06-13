import pytest
from src.orchestrator.phase_resolver import WorkflowPhaseResolver
from src.orchestrator.state import AgentState
from src.orchestrator.state_patch import WorkflowPhase


class TestWorkflowPhaseResolver:
    def test_resolve_returns_existing_phase(self):
        state = AgentState(phase=WorkflowPhase.WRITING)
        assert WorkflowPhaseResolver.resolve(state) == WorkflowPhase.WRITING

    def test_resolve_infers_writing_from_scene_plan_list(self):
        state = AgentState(scene_plan_list=[{"goal": "test"}], scene_text="")
        assert WorkflowPhaseResolver.resolve(state) == WorkflowPhase.WRITING

    def test_resolve_infers_validating_from_validation_result(self):
        state = AgentState(validation_result={"passed": True})
        assert WorkflowPhaseResolver.resolve(state) == WorkflowPhase.VALIDATING

    def test_resolve_infers_planning_from_chapter_and_no_plan(self):
        state = AgentState(current_chapter=1, scene_plan_list=[])
        assert WorkflowPhaseResolver.resolve(state) == WorkflowPhase.PLANNING

    def test_resolve_default_planning(self):
        state = AgentState()
        assert WorkflowPhaseResolver.resolve(state) == WorkflowPhase.PLANNING