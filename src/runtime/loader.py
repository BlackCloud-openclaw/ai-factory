"""
PluginLoader - 从 Manifest 加载 SurfaceDefinition
不操作 Registry，只返回 SurfaceDefinition 元组
"""

import importlib
from typing import Tuple, List
from src.surfaces.definition import SurfaceDefinition


class PluginLoader:
    """
    PluginLoader 只负责加载，不负责存储或查询
    
    职责：
    - 从 Manifest 读取模块名列表
    - 动态导入模块
    - 提取 SurfaceDefinition 对象
    - 返回 Tuple[SurfaceDefinition, ...]
    
    不负责：
    - 注册到 Registry
    - 管理生命周期
    - 冻结或锁定
    """
    
    @classmethod
    def load_from_manifest(cls, manifest_module: str = "src.surfaces.__manifest__") -> Tuple[SurfaceDefinition, ...]:
        """
        从 Manifest 加载所有 Surface
        
        Manifest 应定义 SURFACE_MODULES 列表，包含所有模块名字符串
        
        Returns:
            Tuple[SurfaceDefinition, ...]: 加载的 Surface 列表
        
        Raises:
            RuntimeError: 加载失败时抛出
        """
        try:
            manifest = importlib.import_module(manifest_module)
            module_names: List[str] = getattr(manifest, "SURFACE_MODULES", [])
        except ImportError as e:
            raise RuntimeError(f"Failed to load manifest from {manifest_module}: {e}")
        
        surfaces: List[SurfaceDefinition] = []
        
        for module_name in module_names:
            try:
                module = importlib.import_module(module_name)
            except ImportError as e:
                raise RuntimeError(f"Failed to import surface module {module_name}: {e}")
            
            # 约定：每个模块导出一个名为 {SurfaceName}Surface 的对象
            # 例如 reasoning.py 中定义 ReasoningSurface
            for attr_name in dir(module):
                if attr_name.endswith("Surface"):
                    obj = getattr(module, attr_name)
                    if isinstance(obj, SurfaceDefinition):
                        surfaces.append(obj)
        
        return tuple(surfaces)
    
    @classmethod
    def load_from_list(cls, surfaces: List[SurfaceDefinition]) -> Tuple[SurfaceDefinition, ...]:
        """
        从显式列表加载（用于测试）
        """
        return tuple(surfaces)