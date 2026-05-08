# src/agents/writer.py
import time
from typing import Dict, Any, Optional

from src.agents.base import BaseAgent
from src.orchestrator.state import AgentState
from src.execution.llm_router_pool import get_llm_router_pool
from src.model_router import get_router
from src.config import config
from src.common.logging import setup_logging
from src.common.retry import retry_with_backoff

logger = setup_logging("agents.writer")


class WritingAgent(BaseAgent):
    task_type = "writing"   # 固定任务类型

    async def run(self, state: AgentState) -> Dict[str, Any]:
        logger.info("WritingAgent starting")
        start_time = time.time()

        scene_plan = state.scene_plan or {}
        constraints = state.writing_constraints or {}

        # 从路由器获取写作模型（固定使用 writing 任务类型）
        router = get_router()
        model_name = router.get_model_for_task("writing")
        if not model_name:
            logger.error("No writing model configured")
            return {"scene_text": "[错误] 未配置写作模型", "final_answer": ""}

        prompt = self._build_prompt(scene_plan, constraints)

        pool = get_llm_router_pool()
        try:
            text = await pool.call(
                model_name,
                self._call_llm,
                prompt,
                timeout=getattr(config, 'llm_timeout_writing', 600),
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            text = f"[生成失败: {e}]"

        duration = time.time() - start_time
        logger.info(f"WritingAgent completed, duration={duration:.2f}")
        return {"scene_text": text, "final_answer": text}

    def _build_prompt(self, scene_plan: Dict, constraints: Dict) -> str:
        """构建写作提示词，强调直接输出正文，禁止思考过程"""
        lines = []
        if constraints.get("character_states"):
            lines.append("[当前角色状态]")
            lines.append(str(constraints["character_states"]))
        if constraints.get("must_events"):
            lines.append("[必须发生的事件]")
            lines.append("\n".join(constraints["must_events"]))
        if constraints.get("forbidden_events"):
            lines.append("[禁止事件]")
            lines.append("\n".join(constraints["forbidden_events"]))
        if constraints.get("style_profile"):
            lines.append("[风格要求]")
            lines.append(str(constraints["style_profile"]))
        lines.append("[场景计划]")
        lines.append(f"目标: {scene_plan.get('goal', '')}")
        lines.append(f"冲突: {scene_plan.get('conflict', '')}")
        lines.append(f"结果: {scene_plan.get('outcome', '')}")
        lines.append(f"参与角色: {scene_plan.get('characters', [])}")
        lines.append("\n请根据以上约束写出场景正文（约2000字）。")
        lines.append("最重要规则：严禁输出任何思考过程、分析、计划、括号注释、额外标记。")
        lines.append("直接输出小说正文，从第一句开始就是故事内容。不要输出“场景计划”标题，不要输出“目标”、“冲突”等字段。")
        return "\n".join(lines)

    @retry_with_backoff(max_retries=2, base_delay=1.0)
    async def _call_llm(self, model_name: str, prompt: str, **kwargs) -> str:
        """实际调用 LLM 生成文本，忽略传入的 model_name，使用 task_type 对应的实际模型"""
        from openai import AsyncOpenAI
        from src.model_router import get_router

        # 获取写作任务对应的真实模型名称
        actual_model = get_router().get_model_for_task(self.task_type)

        base_url = kwargs.get('base_url')
        if not base_url:
            pool = get_llm_router_pool()
            base_url = pool.get_base_url(actual_model)

        client = AsyncOpenAI(api_key="not-needed", base_url=base_url)
        response = await client.chat.completions.create(
            model=actual_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=4000,
        )
        message = response.choices[0].message
        content = message.content or ""
        if not content and hasattr(message, "reasoning_content"):
            content = message.reasoning_content or ""

        logger.info(f"Received response from {actual_model}, length={len(content)}")
        if not content:
            logger.warning("Empty response from LLM")
        return content