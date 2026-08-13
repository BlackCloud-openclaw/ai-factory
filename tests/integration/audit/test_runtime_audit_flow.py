# tests/integration/audit/test_runtime_audit_flow.py
"""
Phase 10.4.1: Runtime Audit 端到端集成测试
验证 ControlledWriter → Audit → Report → Store → Reload 完整闭环。
"""

import pytest
import asyncio
from pathlib import Path
import tempfile

from src.writing.audit import (
    audit_writer,
    AuditCoordinator,
    AuditConfig,
    AuditReportStore,
    MemoryPayloadResolver,
    PayloadRef,
    ComprehensiveReport,
)


# ============================================================================
# Mock 组件：模拟 ControlledWriter 行为
# ============================================================================

class MockControlledWriter:
    """
    模拟 ControlledWriter，包含多个阶段（planning, prompt, draft）。
    """

    async def execute(self, novel_id: str, volume: int, chapter: int, scene_idx: int):
        """
        模拟 Writer 执行：生成 Planning → Prompt → Draft 数据。
        """
        # 1. Planning
        planning_data = {
            "goal": "defeat the dragon",
            "conflict": "dragon is immune to fire",
            "outcome": "find a ice weapon",
            "must_events": ["encounter dragon", "discover weakness", "retreat"]
        }

        # 2. Prompt (基于 planning)
        prompt_data = {
            "goal": "defeat the dragon",
            "must_events": ["encounter dragon", "discover weakness"]
        }

        # 3. Draft
        draft_text = "The hero faced the dragon, but his fire spells had no effect..."

        # 4. Result
        result = {
            "novel_id": novel_id,
            "volume": volume,
            "chapter": chapter,
            "scene_idx": scene_idx,
            "planning": planning_data,
            "prompt": prompt_data,
            "draft": draft_text,
        }
        return result


# ============================================================================
# 集成测试
# ============================================================================

class TestRuntimeAuditFlow:

    @pytest.fixture
    def resolver(self):
        return MemoryPayloadResolver()

    @pytest.fixture
    def store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield AuditReportStore(Path(tmpdir))

    @pytest.fixture
    def writer(self):
        return MockControlledWriter()

    # --------------------------------------------------------------------------
    # Case 1: Full Runtime Flow
    # --------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_full_runtime_flow(self, resolver, store, writer):
        """完整闭环：执行 → Trace → Report → Store → Reload。"""

        coordinator = AuditCoordinator(resolver=resolver)
        with coordinator.audit("novel-123", 1, 1, 0) as ctx:
            # 手动记录各阶段（模拟 Writer 执行）
            plan_ref = PayloadRef("memory://planning/001")
            resolver.register(plan_ref, {"goal": "defeat dragon", "outcome": "find ice weapon"})
            plan_id = ctx.collector.record_reference("planning", plan_ref, "digest1", 100)
            ctx.record_stage("planning", outputs={"plan_id": plan_id})

            prompt_ref = PayloadRef("memory://prompt/001")
            resolver.register(prompt_ref, {"goal": "defeat dragon"})
            prompt_id = ctx.collector.record_reference("prompt_bundle", prompt_ref, "digest2", 200)
            ctx.record_stage("prompt", inputs={"plan_id": plan_id}, outputs={"prompt_id": prompt_id})

            draft_ref = PayloadRef("memory://draft/001")
            resolver.register(draft_ref, {"draft": "The hero..."})
            draft_id = ctx.collector.record_reference("draft", draft_ref, "digest3", 300)
            ctx.record_stage("draft", inputs={"prompt_id": prompt_id}, outputs={"draft_id": draft_id})

        # 获取报告
        report = ctx.report
        assert report is not None
        assert report.execution_id == str(ctx.execution_id)
        assert report.summary["total_fields"] > 0

        # 保存报告到 Store
        path = store.save(report, "novel-123")
        assert path.exists()

        # 列出并加载
        entries = store.list(novel_id="novel-123")
        assert len(entries) == 1
        loaded = store.load(entries[0])
        assert loaded is not None
        assert loaded.execution_id == report.execution_id
        assert loaded.summary["total_fields"] == report.summary["total_fields"]

    # --------------------------------------------------------------------------
    # Case 2: Non-invasive Validation
    # --------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_non_invasive(self, resolver, writer):
        """验证 Audit 不影响 Writer 输出。"""

        # 无审计
        async def execute_without_audit(novel_id, volume, chapter, scene_idx):
            return await writer.execute(novel_id, volume, chapter, scene_idx)

        result_without = await execute_without_audit("novel-123", 1, 1, 0)

        # 有审计（使用装饰器）
        @audit_writer(resolver=resolver)
        async def execute_with_audit(novel_id, volume, chapter, scene_idx):
            return await writer.execute(novel_id, volume, chapter, scene_idx)

        result_with = await execute_with_audit("novel-123", 1, 1, 0)

        # 验证输出完全一致
        assert result_without == result_with

    # --------------------------------------------------------------------------
    # Case 3: Failure Isolation
    # --------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_failure_isolation(self, writer):
        """验证 Audit 失败不影响 Writer 执行。"""

        # 创建一个会引发异常的 Resolver
        class BrokenResolver(MemoryPayloadResolver):
            def resolve(self, ref):
                raise RuntimeError("Broken resolver")

        broken_resolver = BrokenResolver()

        @audit_writer(resolver=broken_resolver)
        async def audited_execute(novel_id, volume, chapter, scene_idx):
            return await writer.execute(novel_id, volume, chapter, scene_idx)

        # Writer 应正常执行，不受 Audit 异常影响
        # Coordinator._generate_report 内部捕获了异常，不会向上传播
        result = await audited_execute("novel-123", 1, 1, 0)
        assert result is not None
        assert result["novel_id"] == "novel-123"