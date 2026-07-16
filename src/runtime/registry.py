"""
SurfaceRegistry - 不可变目录（Catalog）实现
构造后即 frozen，无需 register() / freeze() / reset()
"""

from typing import Tuple, Optional
from src.surfaces.definition import SurfaceDefinition
from src.runtime.catalog import SurfaceCatalog
from src.runtime.exceptions import DuplicateSurfaceError


class SurfaceRegistry(SurfaceCatalog):
    """
    SurfaceRegistry 是不可变目录
    
    特点：
    - 构造时接收 SurfaceDefinition 元组
    - 构造时检查重复 ID
    - 构造后即 frozen，不可修改
    - 实现 SurfaceCatalog 协议
    
    用法：
        surfaces = PluginLoader.load_from_manifest()
        registry = SurfaceRegistry(surfaces)
        builder = RuntimeBuilder(registry)
    """
    
    def __init__(self, surfaces: Tuple[SurfaceDefinition, ...]):
        """
        构造不可变 Registry
        
        Args:
            surfaces: SurfaceDefinition 元组
        
        Raises:
            DuplicateSurfaceError: 检测到重复 Surface ID 时抛出
        """
        self._catalog: dict = {}
        
        for surface in surfaces:
            sid = surface.metadata.id
            if sid in self._catalog:
                raise DuplicateSurfaceError(
                    f"Duplicate surface ID '{sid}' detected. "
                    f"First: {self._catalog[sid].metadata.display_name}, "
                    f"Second: {surface.metadata.display_name}"
                )
            self._catalog[sid] = surface
    
    def get(self, id: str) -> Optional[SurfaceDefinition]:
        """按 ID 查询 Surface"""
        return self._catalog.get(id)
    
    def get_all(self) -> Tuple[SurfaceDefinition, ...]:
        """返回所有 Surface（按构造时的顺序）"""
        return tuple(self._catalog.values())
    
    def get_ids(self) -> Tuple[str, ...]:
        """返回所有 Surface ID"""
        return tuple(self._catalog.keys())
    
    def __len__(self) -> int:
        """返回 Surface 数量"""
        return len(self._catalog)
    
    def __contains__(self, id: str) -> bool:
        """检查 Surface 是否存在"""
        return id in self._catalog