# tests/regression/test_phase7_runtime_regression.py

import pytest
import asyncio

from src.workflow.revision_workflow import RevisionWorkflow


def test_runtime_workflow_does_not_raise_snapshot_error():
    """回归测试：验证 RevisionWorkflow 不再因缺少 snapshot 参数而报错"""
    workflow = RevisionWorkflow(
        enable_revision=False,
    )

    draft = "这是一个测试文本。"

    result = asyncio.run(workflow.execute(draft))

    assert "final_text" in result
    assert "compliance" in result
    assert "stages" in result
    assert "artifacts" in result