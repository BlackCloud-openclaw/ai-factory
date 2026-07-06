# src/domain/identity.py
"""
统一角色身份接口 - 所有业务代码从此获取角色信息

使用规范：
1. 业务逻辑使用 ID 进行操作：get_main_character_id()
2. 仅在输出（正文、日志）时使用名称：get_character_name(id)
3. 禁止在业务代码中直接使用角色名字符串
"""

from src.config.character_config import (
    # 配置加载
    load_character_config,
    reload_character_config,

    # 数据类型
    CharacterIdentity,
    ArtifactIdentity,
    CharacterConfig,

    # 主角查询（推荐使用 ID 版本）
    get_main_character_id,
    get_main_character_name,

    # 名称查询
    get_character_name,
    get_character_id_by_name,

    # 道具查询
    get_artifact_name,
    resolve_artifact,
)

# ========== 为兼容性提供别名 ==========
# 部分模块（如 voiceprint.py, state_projection.py）需要 get_character_config
get_character_config = load_character_config

__all__ = [
    # 配置加载
    "load_character_config",
    "reload_character_config",
    "get_character_config",          # 添加此别名

    # 数据类型
    "CharacterIdentity",
    "ArtifactIdentity",
    "CharacterConfig",

    # 主角接口
    "get_main_character_id",
    "get_main_character_name",

    # 角色接口
    "get_character_name",
    "get_character_id_by_name",

    # 道具接口
    "get_artifact_name",
    "resolve_artifact",
]