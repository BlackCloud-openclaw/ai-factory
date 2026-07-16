"""
Phase 7B-3: Dialogue Repair Test
验证 EditCompiler 能从 Snapshot 加载 Repair 策略并生成 EditAction
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime.loader import PluginLoader
from src.runtime.registry import SurfaceRegistry
from src.runtime.builder import RuntimeBuilder
from src.runtime.snapshot import RuntimeConfig
from src.runtime.observation_compiler import ObservationCompiler
from src.runtime.validator import Validator
from src.runtime.edit_compiler import EditCompiler, EditOperation


def test_dialogue_repair():
    print("=" * 60)
    print("Phase 7B-3: Dialogue Repair Test")
    print("=" * 60)

    # 1. 加载 Surface
    surfaces = PluginLoader.load_from_manifest()
    registry = SurfaceRegistry(surfaces)
    builder = RuntimeBuilder(registry)
    config = RuntimeConfig(enabled_surfaces=("reasoning", "dialogue"))
    snapshot = builder.with_config(config).build()

    # 2. 初始化组件
    obs_compiler = ObservationCompiler()
    validator = Validator()
    edit_compiler = EditCompiler()

    # === Case A：无 Dialogue → 应生成 INSERT_DIALOGUE ===
    text_no_dialogue = "林逸沉默地看着窗外。"
    ir = obs_compiler.compile(text_no_dialogue, snapshot)
    report = validator.validate(snapshot, ir)

    # 验证 dialogue 层不合规
    dialogue_result = next((r for r in report.layer_results if r.layer == "dialogue"), None)
    assert dialogue_result is not None
    assert dialogue_result.compliant is False

    # 生成 EditPlan
    plan = edit_compiler.compile_with_snapshot(snapshot, report, text_no_dialogue, ir, diagnosis_id="test_repair")

    # 过滤出 dialogue_marker 相关的 action
    dialogue_actions = [a for a in plan.actions if a.payload_type == "dialogue_marker"]
    assert len(dialogue_actions) > 0, "No dialogue repair action generated"
    action = dialogue_actions[0]

    # 验证 Action
    assert action.operation == EditOperation.INSERT_AFTER
    assert action.anchor_sentence_id == ir.sentences[-1].id
    print("✅ Case A: No dialogue → INSERT_DIALOGUE action generated")
    print(f"   Anchor: {action.anchor_sentence_id} (last sentence)")

    # === Case B：已有 Dialogue → 不应生成 Dialogue Action ===
    text_with_dialogue = "林逸说：「我知道了。」"
    ir2 = obs_compiler.compile(text_with_dialogue, snapshot)
    report2 = validator.validate(snapshot, ir2)

    dialogue_result2 = next((r for r in report2.layer_results if r.layer == "dialogue"), None)
    assert dialogue_result2 is not None
    assert dialogue_result2.compliant is True

    plan2 = edit_compiler.compile_with_snapshot(snapshot, report2, text_with_dialogue, ir2, diagnosis_id="test_repair2")

    # 过滤 dialogue_marker 相关的 action，应该为空
    dialogue_actions2 = [a for a in plan2.actions if a.payload_type == "dialogue_marker"]
    assert len(dialogue_actions2) == 0, "Dialogue repair action generated when dialogue already exists"
    print("✅ Case B: Dialogue exists → no dialogue repair action")

    print("\n" + "=" * 60)
    print("✅ Phase 7B-3: Dialogue Repair passed")
    print("   EditCompiler consumes RuntimeSnapshot")
    print("   Repair strategies from Surface")
    print("   No surface-specific branches in EditCompiler")
    print("=" * 60)


if __name__ == "__main__":
    test_dialogue_repair()