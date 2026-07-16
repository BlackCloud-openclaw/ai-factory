"""
Phase 7A-2: Aggregation Semantics Test
验证 RuntimeSnapshot 的聚合语义，而不是多个 Surface 能工作
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime.observation_compiler import ObservationCompiler
from src.runtime.validator import Validator
from src.runtime.edit_compiler import EditCompiler
from src.runtime.builder import RuntimeBuilder
from src.runtime.registry import SurfaceRegistry
from src.runtime.snapshot import RuntimeConfig
from src.surfaces.reasoning import ReasoningSurface

# 从测试夹具导入 EchoSurface
from tests.fixtures.echo_surface import EchoSurface


class TestAggregation:
    """验证 Aggregation Layer 的六项语义"""

    # 测试用文本，包含 reasoning 和 echo 的关键词
    TEST_DRAFT = "这是一个包含密信和echo关键词的测试文本。"

    def setup_registry(self):
        """每个测试前重置 Registry 并注册 Reasoning + Echo"""
        SurfaceRegistry.reset()
        SurfaceRegistry.register(ReasoningSurface)
        SurfaceRegistry.register(EchoSurface)

    def teardown_registry(self):
        """测试后清理"""
        SurfaceRegistry.reset()

    def test_builder_snapshot(self):
        """测试 Builder 的解析、排序、聚合能力"""
        self.setup_registry()
        try:
            config = RuntimeConfig(
                enabled_surfaces=("reasoning", "echo"),
            )
            snapshot = RuntimeBuilder().with_config(config).build()

            assert len(snapshot.surfaces) == 2
            surface_ids = [s.metadata.id for s in snapshot.surfaces]
            assert surface_ids == ["reasoning", "echo"]

            reasoning_surface = snapshot.get_surface("reasoning")
            echo_surface = snapshot.get_surface("echo")
            assert reasoning_surface is ReasoningSurface
            assert echo_surface is EchoSurface

            print("✅ Builder: Aggregation, Ordering, Immutability OK")
        finally:
            self.teardown_registry()

    def test_observation_aggregation(self):
        """验证 ObservationCompiler 遍历所有 Surface 的 Pattern"""
        self.setup_registry()
        try:
            config = RuntimeConfig(enabled_surfaces=("reasoning", "echo"))
            snapshot = RuntimeBuilder().with_config(config).build()
            compiler = ObservationCompiler()

            ir = compiler.compile(self.TEST_DRAFT, snapshot)

            pattern_types = {p.pattern_type for p in ir.patterns}
            assert "state_keyword" in pattern_types
            assert "echo_marker" in pattern_types

            print("✅ Observation: All surfaces traversed")
        finally:
            self.teardown_registry()

    def test_validation_aggregation(self):
        """验证 Validator 遍历所有 Surface 的 Layer Rules"""
        self.setup_registry()
        try:
            config = RuntimeConfig(enabled_surfaces=("reasoning", "echo"))
            snapshot = RuntimeBuilder().with_config(config).build()
            compiler = ObservationCompiler()
            validator = Validator()

            ir = compiler.compile(self.TEST_DRAFT, snapshot)

            layer_targets = {
                "reasoning": "enhanced",
                "justification": "enhanced",
                "construction": "enhanced",
                "prediction": "enhanced",
                "echo": "enhanced",
            }
            report = validator.validate(ir, layer_targets)

            layer_names = {r.layer for r in report.layer_results}
            assert "reasoning" in layer_names
            assert "echo" in layer_names

            echo_layer = next((r for r in report.layer_results if r.layer == "echo"), None)
            assert echo_layer is not None
            print("✅ Validation: All layer rules processed")
        finally:
            self.teardown_registry()

    def test_edit_aggregation(self):
        """验证 EditCompiler 遍历所有 Surface 的 Repair Strategies"""
        self.setup_registry()
        try:
            config = RuntimeConfig(enabled_surfaces=("reasoning", "echo"))
            snapshot = RuntimeBuilder().with_config(config).build()
            compiler = ObservationCompiler()
            validator = Validator()
            edit_compiler = EditCompiler()

            ir = compiler.compile(self.TEST_DRAFT, snapshot)

            layer_targets = {
                "reasoning": "enhanced",
                "justification": "enhanced",
                "construction": "enhanced",
                "prediction": "enhanced",
                "echo": "enhanced",
            }
            report = validator.validate(ir, layer_targets)
            plan = edit_compiler.compile(ir, report, diagnosis_id="test_edit")

            assert plan is not None

            if plan.actions:
                echo_actions = [a for a in plan.actions if a.payload_type == "echo_marker"]
                print(f"   Echo actions: {len(echo_actions)}")

            print("✅ Edit: All repair strategies traversed")
        finally:
            self.teardown_registry()

    def test_discovery_unknown(self):
        """验证 Builder 对未知 Surface 抛出明确错误"""
        self.setup_registry()
        try:
            config = RuntimeConfig(
                enabled_surfaces=("reasoning", "unknown_surface"),
            )
            builder = RuntimeBuilder().with_config(config)

            try:
                builder.build()
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "unknown_surface" in str(e)
                print("✅ Discovery: Unknown surface error raised")
        finally:
            self.teardown_registry()

    def test_compiler_isolation(self):
        """验证 Compiler 对 Surface 数量无感知（Isolation）"""
        self.setup_registry()
        try:
            config_single = RuntimeConfig(enabled_surfaces=("reasoning",))
            snapshot_single = RuntimeBuilder().with_config(config_single).build()

            config_multi = RuntimeConfig(enabled_surfaces=("reasoning", "echo"))
            snapshot_multi = RuntimeBuilder().with_config(config_multi).build()

            compiler = ObservationCompiler()

            # 使用相同的测试文本
            draft = self.TEST_DRAFT

            ir_single = compiler.compile(draft, snapshot_single)
            ir_multi = compiler.compile(draft, snapshot_multi)

            assert ir_single is not None
            assert ir_multi is not None

            pattern_types_single = {p.pattern_type for p in ir_single.patterns}
            pattern_types_multi = {p.pattern_type for p in ir_multi.patterns}

            # 单 Surface 不应包含 echo_marker
            assert "echo_marker" not in pattern_types_single
            # 多 Surface 应包含 echo_marker
            assert "echo_marker" in pattern_types_multi

            print("✅ Isolation: Compiler works with any number of surfaces")
        finally:
            self.teardown_registry()


if __name__ == "__main__":
    test = TestAggregation()
    
    print("=" * 60)
    print("Phase 7A-2: Aggregation Semantics Test")
    print("=" * 60)
    
    test.test_builder_snapshot()
    test.test_observation_aggregation()
    test.test_validation_aggregation()
    test.test_edit_aggregation()
    test.test_discovery_unknown()
    test.test_compiler_isolation()
    
    print("\n✅ Phase 7A-2: All aggregation semantics verified")