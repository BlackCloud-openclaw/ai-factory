# src/writing/state_compressor.py
import json
import asyncio
from typing import Dict, Any
from src.common.logging import setup_logging
from src.db import get_db_pool

logger = setup_logging("writing.compressor")

async def compress_current_state(novel_id: str, current_state: Dict[str, Any], max_timeline: int = 200, keep_recent: int = 100) -> Dict[str, Any]:
    """压缩状态：截断 timeline，生成旧事件摘要。返回新的 state（浅拷贝）。"""
    if not current_state:
        return current_state
    
    timeline = current_state.get("timeline", [])
    if len(timeline) <= max_timeline:
        return current_state
    
    # 分割：旧部分 + 新部分
    old_part = timeline[:-keep_recent]
    new_part = timeline[-keep_recent:]
    
    # 生成旧部分的摘要（简单方法：拼接前几条 + 后几条，或调用 LLM）
    # 为了性能，我们先简单拼接开头和结尾
    old_summary = f"【旧事件压缩】共 {len(old_part)} 条记录，最早事件：{old_part[0] if old_part else '无'}，... 最末事件：{old_part[-1] if old_part else '无'}"
    # 可选：调用 LLM 生成摘要（耗时，可后台异步）
    # summary = await generate_events_summary(old_part)
    
    # 更新状态
    new_state = current_state.copy()
    new_state["timeline"] = new_part
    new_state["old_events_summary"] = old_summary
    new_state["compressed_at"] = asyncio.get_event_loop().time()
    
    # 持久化到数据库
    pool = get_db_pool()
    if pool:
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE novels SET current_state = $1 WHERE novel_id = $2",
                json.dumps(new_state), novel_id
            )
            logger.info(f"Compressed state for {novel_id}: trimmed from {len(timeline)} to {keep_recent} events")
    return new_state

async def generate_events_summary(events: list) -> str:
    """（可选）调用 LLM 生成事件摘要，用于压缩。"""
    if not events:
        return "无旧事件。"
    # 只取前 5 条和后 5 条作为代表
    sample = events[:5] + (events[-5:] if len(events) > 10 else [])
    text = "事件摘录：\n" + "\n".join(str(e)[:200] for e in sample)
    from openai import AsyncOpenAI
    from src.config import config
    client = AsyncOpenAI(api_key="not-needed", base_url=config.llm_api_url)
    try:
        resp = await client.chat.completions.create(
            model="Qwen3-32B-Q5_K_M",
            messages=[{"role": "user", "content": f"请用一两句话概括以下事件要点：{text}"}],
            temperature=0.2,
            max_tokens=200
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Failed to generate summary: {e}")
        return text[:500]