"""
事件升级器 - 处理 schema 演化

当事件结构发生变化时，通过 upcaster 将旧版本事件升级到新版本，
确保历史数据可读、可重放。
"""
from typing import Dict, Any


class EventUpcaster:
    """事件版本升级器"""
    
    # 当前支持的最新版本
    CURRENT_VERSION = 1
    
    @classmethod
    def upcast(cls, raw_event: Dict[str, Any]) -> Dict[str, Any]:
        """
        将事件升级到最新版本
        
        Args:
            raw_event: 从数据库读取的原始事件数据
            
        Returns:
            升级后的事件数据（最新版本）
        """
        version = raw_event.get("event_version", 1)
        
        # 版本升级链（按顺序调用）
        if version == 1:
            raw_event = cls._upgrade_v1_to_v2(raw_event)
        if version == 2:
            raw_event = cls._upgrade_v2_to_v3(raw_event)
        # 继续添加更多版本...
        
        return raw_event
    
    @classmethod
    def _upgrade_v1_to_v2(cls, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        V1 → V2 升级逻辑
        
        示例：为 RealmUpgradeEvent 添加突破方式字段
        """
        if raw.get("type") == "realm_upgrade":
            if "breakthrough_method" not in raw:
                raw["breakthrough_method"] = "normal"
            if "tribulation_grade" not in raw:
                raw["tribulation_grade"] = None
        
        raw["event_version"] = 2
        return raw
    
    @classmethod
    def _upgrade_v2_to_v3(cls, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        V2 → V3 升级逻辑（示例）
        """
        # 未来可添加：因果标记、临时效果等
        raw["event_version"] = 3
        return raw