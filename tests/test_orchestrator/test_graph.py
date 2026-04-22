import pytest
from src.orchestrator.graph import create_workflow, compile_workflow


class TestWorkflow:
    def test_create_workflow(self):
        workflow = create_workflow()
        assert workflow is not None

    def test_compile_workflow(self):
        compiled = compile_workflow()
        assert compiled is not None

    def test_workflow_has_all_nodes(self):
        workflow = create_workflow()
        # The compiled graph should have the expected nodes
        node_ids = list(workflow.nodes.keys())
        assert "analyze" in node_ids
        assert "research" in node_ids
        assert "code" in node_ids
        assert "validate" in node_ids
