# scripts/verify_phase8.py

"""
Phase 8 快速验证脚本
"""

import sys
from pathlib import Path

# 将项目根目录添加到 sys.path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.capabilities.bootstrap import create_default_registry
from src.runtime.default_runtime import build_default_snapshot
from src.runtime.loader import PluginLoader
from src.runtime.observation_compiler import ObservationCompiler


def verify():
    print("=" * 60)
    print("Phase 8 快速验证")
    print("=" * 60)

    print("\n1. 创建默认 Registry...")
    registry = create_default_registry()
    print(f"   ✅ Registry IDs: {registry.get_all_ids()}")

    print("\n2. 加载默认 Surfaces...")
    surfaces = PluginLoader.load_from_manifest()
    print(f"   ✅ Loaded {len(surfaces)} surfaces")
    for s in surfaces:
        print(f"      - {s.metadata.id}: {s.metadata.display_name}")

    print("\n3. 构建默认 Snapshot...")
    snapshot = build_default_snapshot()
    print(f"   ✅ Snapshot: {snapshot.snapshot_id}")
    print(f"   ✅ Surfaces: {snapshot.get_surface_ids()}")

    print("\n4. 编译测试...")
    compiler = ObservationCompiler()
    ir = compiler.compile("测试文本", snapshot)
    print(f"   ✅ Compiled: {len(ir.sentences)} sentences, {len(ir.patterns)} patterns")

    print("\n" + "=" * 60)
    print("✅ Phase 8 验证通过！")
    print("=" * 60)


if __name__ == "__main__":
    verify()