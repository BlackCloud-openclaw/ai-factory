"""事件 Schema 升级器 - 向后兼容"""
from typing import Dict, Any

LATEST_EVENT_SCHEMA_VERSION = 1  # 当前最新版本


class EventUpcaster:
    @staticmethod
    def upcast(event_data: Dict[str, Any], version: int) -> Dict[str, Any]:
        """将事件数据升级到最新 schema"""
        current = version
        while current < LATEST_EVENT_SCHEMA_VERSION:
            if current == 1:
                event_data = EventUpcaster._upgrade_v1_to_v2(event_data)
            # 未来版本升级在此添加
            # elif current == 2:
            #     event_data = EventUpcaster._upgrade_v2_to_v3(event_data)
            current += 1
        return event_data

    @staticmethod
    def _upgrade_v1_to_v2(event_data: Dict[str, Any]) -> Dict[str, Any]:
        """示例：为旧事件添加 semantic 字段"""
        if 'semantic' not in event_data:
            # 根据事件类型推断默认语义
            event_type = event_data.get('type', '')
            if event_type in ('dialogue', 'discovery'):
                event_data['semantic'] = 'observation'
            elif event_type in ('realm_upgrade', 'item_acquire', 'hp_changed'):
                event_data['semantic'] = 'state_mutation'
            else:
                event_data['semantic'] = 'state_mutation'
        event_data['event_schema_version'] = 2
        return event_data

    # 未来升级函数：
    # @staticmethod
    # def _upgrade_v2_to_v3(event_data: Dict[str, Any]) -> Dict[str, Any]:
    #     ...