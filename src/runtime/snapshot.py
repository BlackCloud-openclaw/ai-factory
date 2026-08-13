# src/runtime/snapshot.py

from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional
from datetime import datetime

from src.surfaces.definition import SurfaceDefinition
from src.capabilities import CapabilityLookup


@dataclass(frozen=True)
class RuntimeConfig:
    """运行时配置 - 决定本次执行的行为"""
    enabled_surfaces: Tuple[str, ...] = field(default_factory=tuple)
    strict_mode: bool = False
    language: str = "zh"
    profile: str = "default"
    diagnostics: bool = True


@dataclass(frozen=True)
class RuntimeMetrics:
    """运行时指标收集器（只读配置）"""
    collect_patterns: bool = True
    collect_timing: bool = True
    collect_layer_scores: bool = True


@dataclass(frozen=True)
class RuntimeSnapshot:
    """
    一次 Runtime 执行开始时的不可变快照

    这是 Compiler 的唯一依赖来源：
    - Compiler 不查 Registry
    - Compiler 不依赖 Config 或 Surfaces 列表以外的对象
    - 所有信息在构造时已解析完成
    """
    snapshot_id: str
    config: RuntimeConfig
    surfaces: Tuple[SurfaceDefinition, ...]
    capability_registry: CapabilityLookup  # 字段名描述对象，类型描述抽象
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    metrics: RuntimeMetrics = field(default_factory=RuntimeMetrics)
    options: Dict[str, Any] = field(default_factory=dict)
    source: str = "builder"

    def get_surface(self, id: str) -> Optional[SurfaceDefinition]:
        """按 ID 查找 Surface（不暴露内部结构）"""
        for surface in self.surfaces:
            if surface.metadata.id == id:
                return surface
        return None

    def get_surface_ids(self) -> Tuple[str, ...]:
        """返回所有 Surface ID"""
        return tuple(s.metadata.id for s in self.surfaces)

    def __len__(self) -> int:
        """返回 Surface 数量"""
        return len(self.surfaces)