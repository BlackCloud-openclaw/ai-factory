"""
Phase 7A-3: Catalog + Loader + Builder Integration Test
验证完整的 Surface Framework 工作流
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime.loader import PluginLoader
from src.runtime.registry import SurfaceRegistry
from src.runtime.builder import RuntimeBuilder
from src.runtime.snapshot import RuntimeConfig
from src.runtime.exceptions import UnknownSurfaceError, DuplicateSurfaceError
from src.surfaces.reasoning import ReasoningSurface


def test_integration():
    """验证完整的 Catalog + Loader + Builder 工作流"""
    print("=" * 60)
    print("Phase 7A-3: Catalog + Loader + Builder Integration")
    print("=" * 60)
    
    # 1. Loader 从 Manifest 加载 Surface
    surfaces = PluginLoader.load_from_manifest()
    print(f"✅ Loaded {len(surfaces)} surfaces from manifest")
    
    # 2. 创建不可变 Registry
    registry = SurfaceRegistry(surfaces)
    print(f"✅ Created immutable registry with {len(registry)} surfaces")
    print(f"   Surface IDs: {registry.get_ids()}")
    
    # 3. Builder 依赖 Catalog 接口
    builder = RuntimeBuilder(registry)
    config = RuntimeConfig(enabled_surfaces=("reasoning",))
    snapshot = builder.with_config(config).build()
    
    print(f"\n✅ Built RuntimeSnapshot: {snapshot.snapshot_id}")
    print(f"   Source: {snapshot.source}")
    print(f"   Surface IDs: {snapshot.get_surface_ids()}")
    print(f"   Surfaces count: {len(snapshot)}")
    
    # 4. 验证 Snapshot 内容
    assert len(snapshot) == 1
    assert snapshot.get_surface_ids() == ("reasoning",)
    assert snapshot.get_surface("reasoning") is ReasoningSurface
    print("\n✅ Snapshot content verified")
    
    # 5. 验证 Builder 对未知 Surface 报错
    config_bad = RuntimeConfig(enabled_surfaces=("reasoning", "unknown"))
    builder_bad = RuntimeBuilder(registry).with_config(config_bad)
    try:
        builder_bad.build()
        assert False, "Should have raised UnknownSurfaceError"
    except UnknownSurfaceError as e:
        print(f"✅ UnknownSurfaceError raised: {e}")
    
    # 6. 验证 Duplicate Detection
    try:
        SurfaceRegistry((ReasoningSurface, ReasoningSurface))
        assert False, "Should have raised DuplicateSurfaceError"
    except DuplicateSurfaceError as e:
        print(f"✅ DuplicateSurfaceError raised: {e}")
    
    # 7. 验证 Registry 不可变性（没有公开的修改方法）
    # Registry 只应有查询方法，不应有 register/add/set 等修改方法
    public_methods = [m for m in dir(registry) if not m.startswith('_')]
    allowed_methods = {'get', 'get_all', 'get_ids', '__len__', '__contains__'}
    unexpected = set(public_methods) - allowed_methods
    if unexpected:
        # 如果有额外的公开方法（比如 register），则失败
        assert False, f"Registry has unexpected public methods: {unexpected}"
    print("✅ Registry is immutable (no public mutation methods)")
    
    print("\n" + "=" * 60)
    print("✅ Phase 7A-3: All checks passed")
    print("=" * 60)


if __name__ == "__main__":
    test_integration()