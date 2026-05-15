# src/writing/summarizer.py
import json
import asyncio
import aiohttp
from typing import Optional, List
from openai import AsyncOpenAI
from src.db import get_db_pool
from src.config import config
from src.common.logging import setup_logging

logger = setup_logging("writing.summarizer")

async def generate_chapter_summary(novel_id: str, volume_num: int, chapter_num: int, chapter_text: str) -> Optional[str]:
    """调用 LLM 生成章节摘要，并存入 chapter_summaries 表。"""
    if not chapter_text or len(chapter_text) < 100:
        logger.warning(f"Chapter text too short for {novel_id} v{volume_num}c{chapter_num}")
        return None

    model = "Qwen3-32B-Q5_K_M"   # 可从配置读取
    base_url = config.llm_api_url
    client = AsyncOpenAI(api_key="not-needed", base_url=base_url)

    prompt = f"""请为以下小说章节生成一个简洁的摘要（200字以内），突出主要情节、关键事件、角色变化和伏笔。

章节内容：
{chapter_text[:4000]}

摘要："""

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
            timeout=30
        )
        summary = response.choices[0].message.content.strip()
        if summary:
            embedding = await generate_embedding(summary)
            pool = get_db_pool()
            if pool:
                async with pool.acquire() as conn:
                    chapter_id = f"{novel_id}_v{volume_num:03d}_c{chapter_num:03d}"
                    await conn.execute("""
                        INSERT INTO chapter_summaries (chapter_id, novel_id, content, embedding)
                        VALUES ($1, $2, $3, $4::vector)
                        ON CONFLICT (chapter_id) DO UPDATE
                        SET content = EXCLUDED.content, embedding = EXCLUDED.embedding, created_at = NOW()
                    """, chapter_id, novel_id, summary, embedding)
                logger.info(f"Saved summary for {chapter_id}")
            return summary
    except Exception as e:
        logger.error(f"Failed to generate summary: {e}")
        return None

async def generate_embedding(text: str) -> str:
    """使用配置的 embedding 服务，返回 pgvector 格式的向量字符串。"""
    endpoint = config.embedding_endpoint
    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(endpoint, json={"input": text}, timeout=timeout) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # 兼容 OpenAI 格式和可能的简化返回值
                    if "data" in data and len(data["data"]) > 0:
                        embedding = data["data"][0].get("embedding")
                        if embedding:
                            return '[' + ','.join(str(x) for x in embedding) + ']'
                    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                        # 某些服务直接返回向量列表
                        embedding = data[0]
                        return '[' + ','.join(str(x) for x in embedding) + ']'
                    else:
                        logger.error(f"Unexpected embedding response format: {data}")
                else:
                    logger.error(f"Embedding service error: {resp.status} {await resp.text()}")
        except Exception as e:
            logger.error(f"Embedding request failed: {e}")

    # 降级：返回零向量
    dim = config.embedding_dim
    return '[' + ','.join(['0.0'] * dim) + ']'

async def retrieve_relevant_summaries(novel_id: str, query_text: str, top_k: int = 3) -> List[str]:
    """根据查询文本，检索最相关的历史章节摘要（向量相似度）"""
    pool = get_db_pool()
    if not pool:
        logger.warning("No database pool, cannot retrieve summaries")
        return []

    query_embedding = await generate_embedding(query_text)
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT content, 1 - (embedding <=> $1::vector) AS similarity
            FROM chapter_summaries
            WHERE novel_id = $2 AND embedding IS NOT NULL
            ORDER BY similarity DESC
            LIMIT $3
        """, query_embedding, novel_id, top_k)
        summaries = [row["content"] for row in rows]
        logger.info(f"Retrieved {len(summaries)} summaries for novel {novel_id}")
        return summaries
    
# src/writing/summarizer.py (在文件末尾添加)
def cosine_similarity(a: list, b: list) -> float:
    """计算两个向量的余弦相似度"""
    import math
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)