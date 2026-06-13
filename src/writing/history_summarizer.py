"""历史摘要生成器 - 为 Writer 提供之前章节的摘要"""

import re
import json
from typing import List, Dict, Any
from src.db import get_db_pool
from src.writing.summarizer import retrieve_relevant_summaries


async def get_chapter_summaries(
    novel_id: str,
    current_volume: int,
    current_chapter: int,
    max_summaries: int = 5,
) -> List[str]:
    """获取最近章节的摘要"""
    pool = get_db_pool()
    if not pool:
        return []
    
    async with pool.acquire() as conn:
        # 获取最近的章节摘要
        rows = await conn.fetch(
            """
            SELECT content, chapter_id
            FROM chapter_summaries
            WHERE novel_id = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            novel_id, max_summaries
        )
        
        summaries = []
        for row in rows:
            content = row["content"]
            chapter_id = row["chapter_id"]
            
            # 从 chapter_id 解析卷号和章号
            # 格式: {novel_id}_v{volume:03d}_c{chapter:03d}
            match = re.search(r'_v(\d+)_c(\d+)', chapter_id)
            if match:
                vol_num = int(match.group(1))
                chap_num = int(match.group(2))
                summaries.append(f"第{vol_num}卷第{chap_num}章摘要：{content[:300]}")
            else:
                summaries.append(content[:300])
        
        return summaries


async def get_key_events_summary(
    novel_id: str,
    current_volume: int,
    current_chapter: int,
    max_events: int = 10,
) -> str:
    """获取关键事件摘要"""
    pool = get_db_pool()
    if not pool:
        return ""
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT event_type, event_data, chapter_num
            FROM narrative_events
            WHERE novel_id = $1 
              AND (volume_num < $2 OR (volume_num = $2 AND chapter_num < $3))
              AND event_type IN ('realm_upgrade', 'item_acquire', 'relationship_change', 'plot_flag_set')
            ORDER BY id DESC
            LIMIT $4
            """,
            novel_id, current_volume, current_chapter, max_events
        )
        
        if not rows:
            return ""
        
        events = []
        for row in rows:
            evt_data = row["event_data"]
            if isinstance(evt_data, str):
                evt_data = json.loads(evt_data)
            event_type = row["event_type"]
            chapter = row["chapter_num"]
            
            if event_type == "realm_upgrade":
                events.append(f"第{chapter}章：{evt_data.get('actor')}突破至{evt_data.get('to_major_realm')}{evt_data.get('to_minor_stage')}层")
            elif event_type == "item_acquire":
                events.append(f"第{chapter}章：获得{evt_data.get('item')}")
            elif event_type == "relationship_change":
                delta = evt_data.get('delta', 0)
                events.append(f"第{chapter}章：{evt_data.get('from_char')}与{evt_data.get('to_char')}关系变化{delta}")
            elif event_type == "plot_flag_set":
                events.append(f"第{chapter}章：触发剧情标记 {evt_data.get('flag')}")
        
        return "【关键事件回顾】\n" + "\n".join(events) if events else ""