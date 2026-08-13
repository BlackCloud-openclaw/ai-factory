# tests/unit/writing/test_controlled_writer_runtime_injection.py

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock

from src.writing.controlled_writer import ControlledWriter
from src.writing.runtime import RuntimeServices
from src.writing.planning_contract import (
    PlanningContract,
    Intent,
    ContractMetadata,
    Execution,
    ExecutionUnit,
)


def build_test_contract() -> PlanningContract:
    """构建最小测试 Contract。"""
    return PlanningContract(
        scene_id="test-scene",
        intent=Intent(
            goal="test goal",
            conflict="test conflict",
            expected_outcome="test outcome",
        ),
        execution=Execution(
            units=[
                ExecutionUnit(id="1", label="action", description="test action"),
            ]
        ),
        metadata=ContractMetadata(chapter=1, scene_index=0),
    )


class TestControlledWriterRuntimeInjection:

    def test_controlled_writer_accepts_runtime_services(self):
        """验证 ControlledWriter 可以接收 RuntimeServices 注入。"""
        services = Mock(spec=RuntimeServices)
        writer = ControlledWriter(
            runtime_services=services,
            api_base="http://test",
            model="test-model",
        )
        assert writer._runtime_services is services

    def test_controlled_writer_without_runtime_services(self):
        """验证不传入 RuntimeServices 时，_runtime_services 为 None。"""
        writer = ControlledWriter(api_base="http://test", model="test-model")
        assert writer._runtime_services is None

    def test_controlled_writer_preserves_original_init_signature(self):
        """验证原有的构造函数参数仍然可用。"""
        writer = ControlledWriter(
            api_base="http://test",
            model="test-model",
            max_retries_per_segment=3,
            enable_fallback=False,
        )
        assert writer.api_base == "http://test"
        assert writer.model == "test-model"
        assert writer.max_retries_per_segment == 3
        assert writer.enable_fallback is False
        assert writer._runtime_services is None

    def test_controlled_writer_with_runtime_services_preserves_other_params(self):
        """验证同时传入 RuntimeServices 和其他参数时都正确保存。"""
        services = Mock(spec=RuntimeServices)
        writer = ControlledWriter(
            api_base="http://custom",
            model="custom-model",
            max_retries_per_segment=5,
            enable_fallback=False,
            runtime_services=services,
        )
        assert writer.api_base == "http://custom"
        assert writer.model == "custom-model"
        assert writer.max_retries_per_segment == 5
        assert writer.enable_fallback is False
        assert writer._runtime_services is services

    @pytest.mark.asyncio
    async def test_execute_does_not_consume_runtime_services(self):
        """
        验证 execute 方法当前不消费 RuntimeServices。
        这是 Phase 11.3.1 的设计决定：只注入但不消费。
        """
        services = Mock(spec=RuntimeServices)
        writer = ControlledWriter(
            runtime_services=services,
            api_base="http://test",
            model="test-model",
        )

        # Mock 内部执行，避免真实 LLM 调用
        writer._execute_segment = AsyncMock(
            return_value=("test text", [], True)
        )

        contract = build_test_contract()
        await writer.execute(contract)

        # 验证 execute 没有调用 audit 相关方法
        services.audit.assert_not_called()
        services.audit_context.assert_not_called()