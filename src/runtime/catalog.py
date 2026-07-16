"""
SurfaceCatalog - 只读目录协议
Builder 的唯一依赖接口
"""

from typing import Protocol, Tuple, Optional
from src.surfaces.definition import SurfaceDefinition


class SurfaceCatalog(Protocol):
    """
    只读目录协议
    
    Builder 通过此接口获取 Surface，而不依赖具体 Registry 实现。
    这允许测试时使用 FakeCatalog，而不需要真实的 Registry。
    """
    
    def get(self, id: str) -> Optional[SurfaceDefinition]:
        """按 ID 查询 Surface"""
        ...
    
    def get_all(self) -> Tuple[SurfaceDefinition, ...]:
        """返回所有 Surface"""
        ...
    
    def get_ids(self) -> Tuple[str, ...]:
        """返回所有 Surface ID"""
        ...