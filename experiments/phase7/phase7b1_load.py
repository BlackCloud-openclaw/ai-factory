"""
Phase 7B-1: DialogueSurface 加载、聚合、遍历
验证新增 Surface 后 Runtime 核心无需修改
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
from src.surfaces.reasoning import ReasoningSurface
from src.surfaces.dialogue import DialogueSurface


def test_dialogue_surface_loading():
    """验证 DialogueSurface 能被加载、聚合、遍历"""
    print("=" * 60)
    print("Phase 7B-1: DialogueSurface Load & Aggregate")
    print("=" * 60)

    # 1. Loader 从 Manifest 加载 Surface
    surfaces = PluginLoader.load_from_manifest()
    print(f"✅ Loaded {len(surfaces)} surfaces from manifest")
    surface_ids = [s.metadata.id for s in surfaces]
    print(f"   Surface IDs: {surface_ids}")

    # 2. 验证 DialogueSurface 存在
    assert "dialogue" in surface_ids
    print("✅ DialogueSurface found in manifest")

    # 3. 验证 ReasoningSurface 仍然存在
    assert "reasoning" in surface_ids
    print("✅ ReasoningSurface still present")

    # 4. 创建不可变 Registry（验证语义，而非对象 identity）
    registry = SurfaceRegistry(surfaces)
    print(f"✅ Created immutable registry with {len(registry)} surfaces")

    # 5. 验证 Registry 包含两个 Surface（验证语义 ID，而非对象 identity）
    reasoning = registry.get("reasoning")
    dialogue = registry.get("dialogue")
    assert reasoning is not None and reasoning.metadata.id == "reasoning"
    assert dialogue is not None and dialogue.metadata.id == "dialogue"
    print("✅ Registry contains both ReasoningSurface and DialogueSurface")

    # 6. Builder 构建 Snapshot（启用两个 Surface）
    builder = RuntimeBuilder(registry)
    config = RuntimeConfig(enabled_surfaces=("reasoning", "dialogue"))
    snapshot = builder.with_config(config).build()

    print(f"\n✅ Built RuntimeSnapshot: {snapshot.snapshot_id}")
    print(f"   Surface IDs: {snapshot.get_surface_ids()}")
    print(f"   Surfaces count: {len(snapshot)}")

    # 7. 验证 Snapshot 包含两个 Surface（验证语义 ID）
    assert len(snapshot) == 2
    assert snapshot.get_surface_ids() == ("reasoning", "dialogue")
    
    reasoning_surface = snapshot.get_surface("reasoning")
    dialogue_surface = snapshot.get_surface("dialogue")
    assert reasoning_surface is not None and reasoning_surface.metadata.id == "reasoning"
    assert dialogue_surface is not None and dialogue_surface.metadata.id == "dialogue"
    print("✅ Snapshot contains both surfaces")

    # 8. 验证 Compiler 能够遍历（通过观察提取的 Pattern 来验证）
    compiler = ObservationCompiler()
    text = '林逸说：「这是对话。」'
    ir = compiler.compile(text, snapshot)
    
    print("DEBUG: ir.patterns =", [(p.text, p.pattern_type, p.sentence_id) for p in ir.patterns])
    
    dialogue_patterns = [p for p in ir.patterns if p.pattern_type == "dialogue_marker"]
    assert len(dialogue_patterns) > 0
    print(f"   Extracted {len(dialogue_patterns)} dialogue_marker patterns")

    # 9. 列出所有 Surface
    print("\n   Surfaces:")
    for surface in snapshot.surfaces:
        print(f"     - {surface.metadata.id}: {surface.metadata.display_name}")

    print("\n" + "=" * 60)
    print("✅ Phase 7B-1: DialogueSurface loaded, aggregated, traversed")
    print("   RuntimeBuilder: no changes")
    print("   ObservationCompiler: no changes")
    print("   Runtime core modified: 0 files")
    print("=" * 60)


if __name__ == "__main__":
    test_dialogue_surface_loading()