# src/runtime/loader.py

"""
PluginLoader - 从 Manifest 加载 SurfaceDefinition
集成 Day 3 兼容层：自动将旧格式（matcher）升级为 CapabilityRef
"""

import importlib
import logging
from typing import Tuple, List

from src.surfaces.definition import SurfaceDefinition
from src.surfaces.compatibility import upgrade_surfaces, SurfaceCompatibilityError

logger = logging.getLogger(__name__)


class PluginLoader:
    """
    PluginLoader 只负责加载，不负责存储或查询
    
    职责：
    - 从 Manifest 读取模块名列表
    - 动态导入模块
    - 提取 SurfaceDefinition 对象
    - **应用兼容层升级（matcher → capability_ref）**
    - 返回 Tuple[SurfaceDefinition, ...]
    
    不负责：
    - 注册到 Registry
    - 管理生命周期
    - 冻结或锁定
    """
    
    @classmethod
    def load_from_manifest(
        cls,
        manifest_module: str = "src.surfaces.__manifest__"
    ) -> Tuple[SurfaceDefinition, ...]:
        """
        从 Manifest 加载所有 Surface，并自动升级到 Phase 8 格式。
        
        Args:
            manifest_module: Manifest 模块路径
            
        Returns:
            Tuple[SurfaceDefinition, ...]: 升级后的 Surface 列表
            
        Raises:
            RuntimeError: 加载或升级失败时抛出
            SurfaceCompatibilityError: 旧格式无法升级时抛出
        """
        # 1. 加载 Manifest
        try:
            manifest = importlib.import_module(manifest_module)
            module_names: List[str] = getattr(manifest, "SURFACE_MODULES", [])
        except ImportError as e:
            raise RuntimeError(f"Failed to load manifest from {manifest_module}: {e}")

        if not module_names:
            logger.warning(f"No SURFACE_MODULES found in {manifest_module}")
            return ()

        # 2. 动态导入所有 Surface 模块
        surfaces: List[SurfaceDefinition] = []
        errors: List[Exception] = []

        for module_name in module_names:
            try:
                module = importlib.import_module(module_name)
            except ImportError as e:
                errors.append(e)
                logger.error(f"Failed to import surface module {module_name}: {e}")
                continue

            # 约定：每个模块导出一个名为 {SurfaceName}Surface 的对象
            found = False
            for attr_name in dir(module):
                if attr_name.endswith("Surface"):
                    obj = getattr(module, attr_name)
                    if isinstance(obj, SurfaceDefinition):
                        surfaces.append(obj)
                        logger.debug(f"Loaded surface: {obj.metadata.id}")
                        found = True
                        break

            if not found:
                logger.warning(f"No SurfaceDefinition found in module {module_name}")

        if errors:
            raise RuntimeError(
                f"Failed to load {len(errors)} surface modules: {[str(e) for e in errors]}"
            )

        if not surfaces:
            logger.warning("No surfaces loaded from manifest")
            return ()

        # 3. ★★★ 应用兼容层升级（matcher → capability_ref）★★★
        # 这是 Day 3 的核心集成点：Loader 是唯一执行升级的地方
        try:
            logger.info(f"Applying compatibility upgrade to {len(surfaces)} surfaces")
            surfaces = list(upgrade_surfaces(tuple(surfaces)))
            logger.info(f"Compatibility upgrade completed")
        except SurfaceCompatibilityError as e:
            logger.error(f"Surface compatibility upgrade failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during compatibility upgrade: {e}")
            raise RuntimeError(f"Surface compatibility upgrade failed: {e}")

        # 4. 验证升级结果（可选但推荐）
        for surface in surfaces:
            for pattern in surface.observation.patterns:
                if pattern.matcher is not None and pattern.capability_ref is None:
                    # 这不应该发生，但如果有，说明升级逻辑有问题
                    logger.warning(
                        f"Surface {surface.metadata.id} pattern '{pattern.name}' "
                        f"still has matcher='{pattern.matcher}' after upgrade. "
                        "This indicates a compatibility bug."
                    )

        logger.info(f"Loaded and upgraded {len(surfaces)} surfaces")
        return tuple(surfaces)

    @classmethod
    def load_from_list(
        cls,
        surfaces: List[SurfaceDefinition]
    ) -> Tuple[SurfaceDefinition, ...]:
        """
        从显式列表加载（用于测试），并应用兼容层升级。
        
        Args:
            surfaces: SurfaceDefinition 列表
            
        Returns:
            Tuple[SurfaceDefinition, ...]: 升级后的 Surface 列表
        """
        if not surfaces:
            return ()

        logger.info(f"Applying compatibility upgrade to {len(surfaces)} surfaces (from list)")
        try:
            upgraded = upgrade_surfaces(tuple(surfaces))
            logger.info(f"Compatibility upgrade completed")
            return upgraded
        except SurfaceCompatibilityError as e:
            logger.error(f"Surface compatibility upgrade failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during compatibility upgrade: {e}")
            raise RuntimeError(f"Surface compatibility upgrade failed: {e}")