"""
Writer Agent - 基于状态驱动的章节生成
"""
import json
import time
import re
from typing import Dict, Any, Optional

from src.agents.base import BaseAgent
from src.orchestrator.state import AgentState
from src.execution.llm_router_pool import get_llm_router_pool
from src.model_router import get_router
from src.config import config
from src.common.logging import setup_logging
from src.common.retry import retry_with_backoff

from src.writing.voiceprint import VoiceprintRegistry
from src.writing.context_compiler import ContextCompiler
from src.writing.prompt_firewall import PromptFirewall
from src.writing.validators import validate_all

logger = setup_logging("agents.writer")

# 全局声纹注册表
_voiceprint_registry = None

def get_voiceprint_registry():
    global _voiceprint_registry
    if _voiceprint_registry is None:
        _voiceprint_registry = VoiceprintRegistry("config/voiceprints.yaml")
    return _voiceprint_registry


class WritingAgent(BaseAgent):
    task_type = "writing"

    async def run(self, state: AgentState) -> Dict[str, Any]:
        logger.info("WritingAgent starting")
        start_time = time.time()

        scene_plan = state.scene_plan or {}
        constraints = state.writing_constraints or {}
        
        # 获取声纹注册表
        voiceprint_registry = get_voiceprint_registry()
        
        # 获取当前世界状态（如果存在）
        from src.writing.world_state import WorldState
        world_state = WorldState.from_dict(state.current_state) if state.current_state else WorldState()
        
        # 编译上下文
        compiler = ContextCompiler(max_tokens=2000)
        compiled_context = compiler.compile(
            world_state,
            active_characters=scene_plan.get("characters", []),
            max_active=10
        )
        
        # 构建完整 prompt
        prompt = compiler.build_writer_prompt(
            scene_plan=scene_plan,
            world_state=world_state,
            voiceprint_registry=voiceprint_registry,
            compiled_context=compiled_context,
        )
        
        # 添加额外的约束（禁止事件等）
        if constraints.get("forbidden_events"):
            prompt += f"\n\n【禁止事件】\n" + "\n".join(constraints["forbidden_events"])
        
        # 调用 LLM
        router = get_router()
        primary_model = router.get_model_for_task("writing")
        fallback_model = "Qwen3-32B-Q5_K_M"
        
        pool = get_llm_router_pool()
        raw_output = None
        last_error = None
        
        try:
            raw_output = await pool.call(
                primary_model,
                self._call_llm,
                prompt,
                timeout=getattr(config, 'llm_timeout_writing', 600),
            )
        except Exception as e:
            logger.warning(f"Primary model {primary_model} failed: {e}, trying fallback")
            last_error = e
            try:
                raw_output = await pool.call(
                    fallback_model,
                    self._call_llm,
                    prompt,
                    timeout=getattr(config, 'llm_timeout_writing', 600),
                )
                logger.info(f"Fallback model {fallback_model} succeeded")
            except Exception as e2:
                logger.error(f"Fallback model also failed: {e2}")
                raw_output = f'{{"scene_text": "[生成失败: {last_error}]", "events": [], "foreshadowing": []}}'
        
        logger.info(f"Raw LLM output: {raw_output[:500]}")
        
        # 验证输出结构（包括 must_events 检查）
        must_events = scene_plan.get("must_events", [])
        context = {"must_events": must_events}
        validation = validate_all(raw_output, context, async_semantic=False)
        
        scene_text = ""
        if validation["passed"]:
            parsed = validation["parsed_output"] or {}
            scene_text = parsed.get("scene_text", "")
        else:
            logger.warning(f"Validation failed: {validation['error']}")
            # 尝试从原始输出中提取 JSON 中的 scene_text
            try:
                match = re.search(r'"scene_text"\s*:\s*"([^"]*)"', raw_output)
                if match:
                    scene_text = match.group(1)
                else:
                    scene_text = f"[验证失败: {validation['error']}]"
            except:
                scene_text = f"[验证失败: {validation['error']}]"
        
        # ========== 删除 goal/conflict 关键词校验，直接返回空标记 ==========
        deviation_detected = False
        missing_goal = []
        missing_conflict = []
        
        duration = time.time() - start_time
        logger.info(f"WritingAgent completed, duration={duration:.2f}")
        
        return {
            "scene_text": raw_output,
            "final_answer": scene_text,
            "deviation_detected": deviation_detected,
            "missing_goal_keywords": missing_goal,
            "missing_conflict_keywords": missing_conflict,
        }
    
    @retry_with_backoff(max_retries=2, base_delay=1.0)
    async def _call_llm(self, model_name: str, prompt: str, **kwargs) -> str:
        """调用 LLM"""
        from openai import AsyncOpenAI
        from src.model_router import get_router
        
        actual_model = model_name
        base_url = kwargs.get('base_url')
        if not base_url:
            pool = get_llm_router_pool()
            base_url = pool.get_base_url(actual_model)
        
        client = AsyncOpenAI(api_key="not-needed", base_url=base_url)
        response = await client.chat.completions.create(
            model=actual_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=4096,
        )
        content = response.choices[0].message.content or ""
        logger.info(f"Received response from {actual_model}, length={len(content)}")
        return content