# tests/contract/test_runtime_behavior_contract.py

import pytest
from unittest.mock import MagicMock, patch
import asyncio

from src.workflow.revision_workflow import RevisionWorkflow
from src.runtime import RuntimeSnapshot


class TestRuntimeBehaviorContract:
    """Runtime 行为契约测试"""

    def test_revision_workflow_passes_snapshot_to_compiler(self):
        # 创建更完整的 Mock Snapshot
        mock_snapshot = MagicMock(spec=RuntimeSnapshot)
        mock_snapshot.get_surface_ids.return_value = ("reasoning",)
        # ✅ 添加 surfaces 属性（RevisionWorkflow 需要遍历它）
        mock_snapshot.surfaces = ()
        # ✅ 添加 snapshot_id（用于日志）
        mock_snapshot.snapshot_id = "test_snapshot"

        workflow = RevisionWorkflow(
            snapshot=mock_snapshot,
            enable_revision=False,
        )

        with patch.object(workflow._obs_compiler, 'compile') as mock_compile:
            mock_compile.return_value = MagicMock()
            asyncio.run(workflow.execute("test draft"))

            mock_compile.assert_called_once()
            call_args = mock_compile.call_args
            assert call_args[0][0] == "test draft"
            assert call_args[0][1] == mock_snapshot

    def test_revision_workflow_passes_snapshot_to_validator(self):
        # 创建更完整的 Mock Snapshot
        mock_snapshot = MagicMock(spec=RuntimeSnapshot)
        mock_snapshot.get_surface_ids.return_value = ("reasoning",)
        mock_snapshot.surfaces = ()
        mock_snapshot.snapshot_id = "test_snapshot"

        workflow = RevisionWorkflow(
            snapshot=mock_snapshot,
            enable_revision=False,
        )

        with patch.object(workflow._validator, 'validate') as mock_validate:
            # 需要 mock validate 的返回值，因为后续会访问 result.overall_compliance
            mock_report = MagicMock()
            mock_report.overall_compliance = 1.0
            mock_validate.return_value = mock_report

            asyncio.run(workflow.execute("test draft"))

            mock_validate.assert_called_once()
            call_args = mock_validate.call_args
            assert call_args[0][0] == mock_snapshot