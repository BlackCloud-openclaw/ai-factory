# src/runtime/default_runtime.py

from typing import Optional, Tuple

from src.capabilities import CapabilityLookup
from src.capabilities.bootstrap import create_default_registry
from src.runtime.builder import RuntimeBuilder
from src.runtime.loader import PluginLoader
from src.runtime.registry import SurfaceRegistry
from src.runtime.snapshot import RuntimeSnapshot, RuntimeConfig


def build_default_snapshot(
    registry: Optional[CapabilityLookup] = None,
    surfaces: Optional[Tuple] = None,
    config: Optional[RuntimeConfig] = None,
) -> RuntimeSnapshot:
    """
    构建默认的 RuntimeSnapshot。

    这是 Composition Root 的默认实现：
    - 依赖可注入，支持测试和自定义
    - 默认从 Capability 子系统和 Surface 子系统获取

    Args:
        registry: CapabilityLookup，默认使用 create_default_registry()
        surfaces: SurfaceDefinition 元组，默认使用 PluginLoader.load_from_manifest()
        config: RuntimeConfig，默认启用 ("reasoning",)
    """
    if registry is None:
        registry = create_default_registry()

    if surfaces is None:
        surfaces = PluginLoader.load_from_manifest()

    if config is None:
        config = RuntimeConfig(
            enabled_surfaces=("reasoning",),
            # diagnostics 默认 False，由调用方按需开启
        )

    surface_registry = SurfaceRegistry(surfaces)
    builder = RuntimeBuilder(surface_registry, registry)

    return builder.with_config(config).build()