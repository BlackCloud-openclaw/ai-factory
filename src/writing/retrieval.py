"""
叙事事件检索 - 支持向量检索和相关事件召回
"""
from typing import List, Dict, Any, Optional
from .event_store import NarrativeEventStore
from .world_state import WorldState
from .events import NarrativeEvent, event_from_dict


class NarrativeRetriever:
    """叙事事件检索器（简易版，后续可升级向量检索）"""
    
    def __init__(self, event_store: NarrativeEventStore):
        self.event_store = event_store
    
    async def retrieve_relevant_events(
        self,
        novel_id: str,
        query: str,
        current_world: WorldState,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        检索与当前场景相关的事件
        
        简易实现：基于关键词匹配（后续可升级为向量检索）
        """
        # 提取查询中的关键词
        keywords = set(query.split())
        # 添加活跃角色名
        active_chars = current_world.get_active_characters(max_count=10)
        keywords.update(active_chars)
        
        # 获取最近事件（简化：取最近 200 条）
        events = await self.event_store.get_events_since(novel_id, since_event_id=0, limit=200)
        
        # 计算相关性分数（简单词频重叠）
        scored = []
        for evt in events:
            evt_text = str(evt.model_dump())
            score = sum(1 for kw in keywords if kw in evt_text)
            if score > 0:
                scored.append((score, evt))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        top_events = [e.model_dump() for _, e in scored[:top_k]]
        return top_events