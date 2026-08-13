# src/runtime/builder.py

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import uuid

from src.runtime.catalog import SurfaceCatalog
from src.runtime.snapshot import RuntimeSnapshot, RuntimeConfig, RuntimeMetrics
from src.runtime.exceptions import UnknownSurfaceError, SnapshotBuildError
from src.surfaces.definition import SurfaceDefinition
from src.capabilities import CapabilityLookup


class RuntimeBuilder:
    """
    RuntimeBuilder 负责 Composition：
    - 依赖 SurfaceCatalog 和 CapabilityLookup（均为 Protocol）
    - 解析启用的 Surface（优先使用 RuntimeConfig）
    - 确定执行顺序
    - 冻结为 RuntimeSnapshot
    """

    def __init__(
        self,
        catalog: SurfaceCatalog,
        capability_registry: CapabilityLookup,  # 依赖 Protocol
    ):
        self._catalog = catalog
        self._capability_registry = capability_registry
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
        2. 确定启用哪些 Surface
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
            ordered_surfaces.append(surface)

        # 5. 生成 Snapshot ID
        snapshot_id = f"snapshot_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        # 6. 构建不可变 Snapshot
        return RuntimeSnapshot(
            snapshot_id=snapshot_id,
            config=self._config,
            surfaces=tuple(ordered_surfaces),
            capability_registry=self._capability_registry,  # 注入 Registry
            metrics=self._metrics,
            options=self._options.copy(),
            source=self._source,
        )

    # ---- 快捷方法 ----

    @classmethod
    def from_surfaces(
        cls,
        catalog: SurfaceCatalog,
        capability_registry: CapabilityLookup,
        *surface_ids: str,
        config: Optional[RuntimeConfig] = None,
    ) -> RuntimeSnapshot:
        """快速构建：直接指定 Surface ID"""
        builder = cls(catalog, capability_registry)
        for id in surface_ids:
            builder.enable_surface(id)
        if config:
            builder.with_config(config)
        return builder.build()