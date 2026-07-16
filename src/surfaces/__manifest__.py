"""
Surface Manifest - 显式声明所有可用 Surface 的模块名
PluginLoader 负责动态加载
"""

SURFACE_MODULES = [
    "src.surfaces.reasoning",
    "src.surfaces.dialogue",  # Phase 7B 新增
]