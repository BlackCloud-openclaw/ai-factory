# src/writing/narrative_projection.py

"""
Narrative Projection Layer
探索期：宽记录，不预设最终模型

职责：
1. Event → Narrative Projection (LLM)
2. 持久化到 narrative_projection_snapshots
3. 提供读取接口给 Planner
"""
print("===== NARRATIVE_PROJECTION MODULE LOADED =====", flush=True)
import sys
sys.stderr.write("===== NARRATIVE_PROJECTION MODULE LOADED =====\n")
sys.stderr.flush()

import json
import logging
import uuid
import sys
from datetime import datetime
from typing import Dict, Any, Optional

from src.db import get_db_pool
from src.common.prompt_logger import log_prompt
from src.model_router import get_router
from src.execution.llm_router_pool import get_llm_router_pool

# 获取模块级 logger
logger = logging.getLogger(__name__)


PROJECTION_PROMPT = """
你是一位故事分析师。阅读当前事件与已有叙事状态后，分析当前故事的叙事状态。

当前事件：
{event}

已有叙事状态（最近一次投影）：
{last_projection}

请回答以下问题：

1. 当前故事最重要的未完成事项是什么？（focus）
   - subject: 主题是什么（人物/物品/秘密/目标）
   - type: person | item | secret | goal | threat
   - why_matters: 为什么这件事对故事重要

2. 当前正在推进什么过程？（loop）
   - description: 描述这个未完成的过程
   - initiator: 谁在推动它
   - urgency: 0-1，紧迫度

3. 角色当前最关注什么？（attention）
   - target: 关注对象
   - intensity: 0-1，关注强度

4. 当前驱动读者阅读的核心问题是什么？（question）
   - text: 问题文本
   - scope: chapter | arc

5. 如果下一章不推进当前最重要的未完成事项，会产生什么损失？

6. 你对本次分析的信心有多少？（0-1）

输出 JSON，使用以下 Schema：
{{
  "focus": {{"subject": "...", "type": "...", "why_matters": "..."}},
  "loop": {{"description": "...", "initiator": "...", "urgency": 0.0}},
  "attention": {{"target": "...", "intensity": 0.0}},
  "question": {{"text": "...", "scope": "chapter"}},
  "raw_analysis": "...",
  "confidence": 0.0
}}

只输出 JSON，不要有任何额外文本。
"""


class NarrativeProjection:
    """叙事投影数据容器"""

    def __init__(self, data: Dict[str, Any]):
        self.version = 1
        self.source_event_id = data.get("source_event_id")
        self.source_chapter = data.get("source_chapter")
        self.focus = data.get("focus", {})
        self.loop = data.get("loop", {})
        self.attention = data.get("attention", {})
        self.question = data.get("question", {})
        self.raw_analysis = data.get("raw_analysis", "")
        self.confidence = data.get("confidence", 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "projection_version": self.version,
            "source_event_id": self.source_event_id,
            "source_chapter": self.source_chapter,
            "focus": self.focus,
            "loop": self.loop,
            "attention": self.attention,
            "question": self.question,
            "raw_analysis": self.raw_analysis,
            "confidence": self.confidence,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NarrativeProjection":
        return cls(data)


class NarrativeProjector:
    """叙事投影器：Event → Narrative State"""

    @staticmethod
    async def project(
        novel_id: str,
        event: Dict[str, Any],
        chapter: int,
        event_id: int,
        last_projection: Optional[Dict[str, Any]] = None
    ) -> Optional[NarrativeProjection]:
        """
        执行叙事投影
        """
        # 强日志：使用 print 确保输出
        print(f"[NARRATIVE PROJECTION] PROJECT CALLED: novel={novel_id}, chapter={chapter}, event_id={event_id}", flush=True)
        logger.info(f"[NarrativeProjector] PROJECT CALLED: novel={novel_id}, chapter={chapter}, event_id={event_id}")

        try:
            # 1. 构建 Prompt
            prompt = PROJECTION_PROMPT.format(
                event=json.dumps(event, ensure_ascii=False, indent=2, default=str),
                last_projection=json.dumps(last_projection or {}, ensure_ascii=False, indent=2, default=str)
            )

            # 2. 记录 Prompt 日志
            log_prompt("narrative_projection", prompt, metadata={
                "novel_id": novel_id,
                "chapter": chapter,
                "event_id": event_id
            })

            # 3. 调用 LLM
            router = get_router()
            model = router.get_model_for_task("plan")
            pool = get_llm_router_pool()

            async def _call(model_name: str, **kwargs) -> str:
                from openai import AsyncOpenAI
                base_url = pool.get_base_url(model_name)
                client = AsyncOpenAI(api_key="not-needed", base_url=base_url)
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=1024,
                )
                return response.choices[0].message.content or ""

            raw_output = await pool.call(model, _call, timeout=600, agent="narrative_projection")
            print(f"[NARRATIVE PROJECTION] LLM response length: {len(raw_output)}")
            logger.info(f"LLM response received, length={len(raw_output)}")

            # 4. 解析 JSON
            import re
            match = re.search(r'\{.*\}', raw_output, re.DOTALL)
            if not match:
                print(f"[NARRATIVE PROJECTION] ERROR: Failed to extract JSON")
                logger.error(f"Failed to extract JSON from projection output: {raw_output[:200]}")
                return None

            data = json.loads(match.group())
            print(f"[NARRATIVE PROJECTION] JSON parsed successfully")

            # 5. 添加元数据
            data["source_event_id"] = event_id
            data["source_chapter"] = chapter

            projection = NarrativeProjection(data)

            # 6. 持久化 - 添加详细日志
            print(f"[NARRATIVE PROJECTION] About to call _save for event {event_id}")
            logger.info(f"About to call _save for event {event_id}")
            saved = await NarrativeProjector._save(novel_id, chapter, event_id, projection)
            print(f"[NARRATIVE PROJECTION] _save returned: {saved}")
            logger.info(f"_save returned: {saved}")

            if saved:
                print(f"[NARRATIVE PROJECTION] SUCCESS: saved for chapter {chapter}, event {event_id}")
                logger.info(f"Narrative projection saved for chapter {chapter}, event {event_id}")
            else:
                print(f"[NARRATIVE PROJECTION] FAILED: NOT saved for chapter {chapter}, event {event_id}")
                logger.error(f"Narrative projection NOT saved for chapter {chapter}, event {event_id}")

            return projection

        except Exception as e:
            print(f"[NARRATIVE PROJECTION] EXCEPTION: {type(e).__name__}: {e}")
            logger.error(f"Narrative projection failed: {e}", exc_info=True)
            return None

    @staticmethod
    async def _save(novel_id: str, chapter: int, event_id: int, projection: NarrativeProjection) -> bool:
        """保存投影到数据库，返回是否成功"""
        # 强日志
        print(f"[NARRATIVE PROJECTION _save] ENTER: novel={novel_id}, chapter={chapter}, event_id={event_id}")
        logger.info(f"_save ENTER: novel={novel_id}, chapter={chapter}, event_id={event_id}")

        pool = get_db_pool()
        if not pool:
            print(f"[NARRATIVE PROJECTION _save] ERROR: Database pool not available")
            logger.error(f"Database pool not available, projection not saved for chapter {chapter}")
            return False

        print(f"[NARRATIVE PROJECTION _save] Pool acquired")
        logger.info("Pool acquired")

        projection_id = f"proj_{uuid.uuid4().hex[:12]}"
        json_data = projection.to_json()
        print(f"[NARRATIVE PROJECTION _save] JSON data length: {len(json_data)}")

        try:
            async with pool.acquire() as conn:
                print(f"[NARRATIVE PROJECTION _save] Executing INSERT...")
                await conn.execute(
                    """
                    INSERT INTO narrative_projection_snapshots
                    (id, novel_id, chapter, event_id, projection_data)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    projection_id,
                    novel_id,
                    chapter,
                    event_id,
                    json_data
                )
            print(f"[NARRATIVE PROJECTION _save] INSERT success")
            logger.info(f"Saved projection for chapter {chapter}, event {event_id} (id={projection_id})")
            return True
        except Exception as e:
            print(f"[NARRATIVE PROJECTION _save] EXCEPTION: {type(e).__name__}: {e}")
            logger.error(f"Failed to save projection: {e}", exc_info=True)
            return False

    @staticmethod
    async def get_latest(novel_id: str) -> Optional[NarrativeProjection]:
        """获取最新的叙事投影（供 Planner 使用）"""
        pool = get_db_pool()
        if not pool:
            return None

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT projection_data
                FROM narrative_projection_snapshots
                WHERE novel_id = $1
                ORDER BY created_at DESC
                LIMIT 1
                """,
                novel_id
            )

        if not row:
            return None

        data = json.loads(row["projection_data"])
        return NarrativeProjection.from_dict(data)