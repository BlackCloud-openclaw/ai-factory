# src/writing/causality/upcaster.py
"""
事件 Schema 升级器 - 支持版本链升级整个事件信封

设计原则：
- 升级整个事件信封（envelope），不仅仅是 payload
- 版本链式升级，支持多步迁移
- 保持确定性、可逆性（至少向前）
- 禁止修改已存在的必需字段语义
"""
from typing import Dict, Any

# 当前最新 schema 版本
LATEST_EVENT_SCHEMA_VERSION = 2

# 版本链升级函数映射：(from_version, to_version) -> function
UPCASTERS = {}


def register_upcaster(from_ver: int, to_ver: int):
    """装饰器：注册升级函数"""
    def decorator(func):
        UPCASTERS[(from_ver, to_ver)] = func
        return func
    return decorator


@register_upcaster(1, 2)
def upgrade_v1_to_v2(envelope: Dict[str, Any]) -> Dict[str, Any]:
    """
    V1 → V2 升级逻辑：
    - 为 realm_upgrade 事件添加 breakthrough_method 字段（默认 "normal"）
    - 确保 semantic 字段存在（根据事件类型推断）
    - 增加 event_schema_version 字段
    """
    data = envelope.copy()
    event_type = data.get("type")
    payload = data.get("payload", {})

    # 1. 为 realm_upgrade 添加突破方式
    if event_type == "realm_upgrade" and "breakthrough_method" not in payload:
        payload["breakthrough_method"] = "normal"
        data["payload"] = payload

    # 2. 推断 semantic（如果缺失）
    if "semantic" not in data:
        if event_type in ("dialogue", "discovery", "observation"):
            data["semantic"] = "observation"
        elif event_type in ("realm_upgrade", "item_acquire", "hp_changed", "mp_changed", "location_enter"):
            data["semantic"] = "state_mutation"
        else:
            data["semantic"] = "state_mutation"

    # 3. 更新版本号
    data["event_schema_version"] = 2
    return data


def upcast_event_envelope(envelope: Dict[str, Any]) -> Dict[str, Any]:
    """
    将事件信封升级到最新 schema。
    如果版本已是最新，直接返回原对象（不复制）。
    """
    current_version = envelope.get("event_schema_version", 1)
    if current_version >= LATEST_EVENT_SCHEMA_VERSION:
        return envelope

    data = envelope.copy()  # 避免修改原始数据
    version = current_version
    while version < LATEST_EVENT_SCHEMA_VERSION:
        next_version = version + 1
        key = (version, next_version)
        if key in UPCASTERS:
            data = UPCASTERS[key](data)
        else:
            # 如果没有定义升级函数，直接提升版本号（谨慎）
            data["event_schema_version"] = next_version
        version = next_version
    return data


def upcast_event_data(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    兼容旧接口：直接调用 upcast_event_envelope。
    """
    return upcast_event_envelope(event_data)