"""
RuntimeBuilder - 组合层（Composition Layer）
负责 resolve → order → freeze → snapshot
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import uuid

from src.runtime.catalog import SurfaceCatalog
from src.runtime.snapshot import RuntimeSnapshot, RuntimeConfig, RuntimeMetrics
from src.runtime.exceptions import UnknownSurfaceError, SnapshotBuildError
from src.surfaces.definition import SurfaceDefinition
from src.runtime.registry import SurfaceRegistry


class RuntimeBuilder:
    """
    RuntimeBuilder 负责 Composition：
    - 依赖 SurfaceCatalog 接口（而非具体 Registry）
    - 解析启用的 Surface（优先使用 RuntimeConfig）
    - 确定执行顺序
    - 冻结为 RuntimeSnapshot
    """
    
    def __init__(self, catalog: SurfaceCatalog):
        self._catalog = catalog
        self._config = RuntimeConfig()
        self._metrics = RuntimeMetrics()
        self._options: Dict[str, Any] = {}
        self._source: str = "builder"
        self._surface_ids: List[str] = []
        self._custom_order: Optional[List[str]] = None
    
    # ---- 配置方法 ----
    
    def with_config(self, config: RuntimeConfig) -> "RuntimeBuilder":
        """设置 RuntimeConfig"""
        self._config = config
        return self
    
    def with_metrics(self, metrics: RuntimeMetrics) -> "RuntimeBuilder":
        """设置 RuntimeMetrics"""
        self._metrics = metrics
        return self
    
    def with_options(self, options: Dict[str, Any]) -> "RuntimeBuilder":
        """设置扩展选项"""
        self._options = options
        return self
    
    def with_source(self, source: str) -> "RuntimeBuilder":
        """设置来源标识（仅用于调试）"""
        self._source = source
        return self
    
    def enable_surface(self, id: str) -> "RuntimeBuilder":
        """启用一个 Surface（补充到 _surface_ids）"""
        if id not in self._surface_ids:
            self._surface_ids.append(id)
        return self
    
    def enable_surfaces(self, ids: List[str]) -> "RuntimeBuilder":
        """启用多个 Surface（补充到 _surface_ids）"""
        for id in ids:
            if id not in self._surface_ids:
                self._surface_ids.append(id)
        return self
    
    def with_order(self, ids: List[str]) -> "RuntimeBuilder":
        """显式指定执行顺序（覆盖默认顺序）"""
        self._custom_order = ids
        return self
    
    # ---- 核心方法 ----
    
    def build(self) -> RuntimeSnapshot:
        """
        构建 RuntimeSnapshot
        
        流程：
        1. 从 Catalog 获取所有 Surface
        2. 确定启用哪些 Surface：
           - 优先使用 RuntimeConfig.enabled_surfaces
           - 如果 config 为空，使用 Builder 的 _surface_ids
           - 如果两者都为空，启用所有 Surface
        3. 检查未知 Surface
        4. 确定执行顺序
        5. 冻结为 Snapshot
        """
        all_surfaces = {s.metadata.id: s for s in self._catalog.get_all()}
        
        # 1. 确定启用的 Surface ID（优先使用 config）
        config_enabled = self._config.enabled_surfaces
        if config_enabled:
            enabled_ids = list(config_enabled)
        elif self._surface_ids:
            enabled_ids = self._surface_ids
        else:
            # 默认：启用所有 Surface
            enabled_ids = list(all_surfaces.keys())
        
        # 2. 检查未知 Surface
        for id in enabled_ids:
            if id not in all_surfaces:
                raise UnknownSurfaceError(f"Unknown surface: {id}")
        
        # 3. 确定执行顺序
        if self._custom_order:
            order = [id for id in self._custom_order if id in enabled_ids]
            # 补充未在自定义顺序中的 Surface
            for id in enabled_ids:
                if id not in order:
                    order.append(id)
        else:
            order = enabled_ids
        
        # 4. 获取 SurfaceDefinition 并按顺序排列
        ordered_surfaces = []
        for id in order:
            surface = all_surfaces.get(id)
            if surface is None:
                raise UnknownSurfaceError(f"Surface '{id}' not found in catalog")
            ordered_surfaces.append(self._freeze_surface(surface))
        
        # 5. 生成 Snapshot ID
        snapshot_id = f"snapshot_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # 6. 构建不可变 Snapshot
        return RuntimeSnapshot(
            snapshot_id=snapshot_id,
            config=self._config,
            surfaces=tuple(ordered_surfaces),
            metrics=self._metrics,
            options=self._options.copy(),
            source=self._source,
        )
    
    # ---- 内部方法 ----
    
    def _freeze_surface(self, surface: SurfaceDefinition) -> SurfaceDefinition:
        """
        Phase 7A: 直接返回 SurfaceDefinition
        Phase 8: 可升级为 CompiledSurface
        """
        return surface
    
    # ---- 快捷方法 ----
    
    @classmethod
    def from_surfaces(cls, catalog: SurfaceCatalog, *surface_ids: str, config: Optional[RuntimeConfig] = None) -> RuntimeSnapshot:
        """快速构建：直接指定 Surface ID"""
        builder = cls(catalog)
        for id in surface_ids:
            builder.enable_surface(id)
        if config:
            builder.with_config(config)
        return builder.build()
    
# ========== 模块级函数（放在类外部） ==========
def build_default_snapshot(
    surface_ids: Tuple[str, ...] = ("reasoning",)
) -> RuntimeSnapshot:
    """
    构建默认的 RuntimeSnapshot。
    
    这是 Workflow 获取 Snapshot 的统一入口。
    如果未来引入统一工厂，只需修改此函数内部实现。
    """
    from src.surfaces.reasoning import ReasoningSurface
    registry = SurfaceRegistry((ReasoningSurface,))
    config = RuntimeConfig(enabled_surfaces=surface_ids)
    return RuntimeBuilder(registry).with_config(config).build()