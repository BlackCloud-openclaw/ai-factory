# src/agents/rewrite.py
"""
Rewrite v2 - Drama Amplifier

定位：不创造戏剧结构，而是放大 Writer 已经生成的戏剧性。
职责：
- Layer 1 (20%)：基础清洁（删除禁止模式、合并短句、调整节奏）
- Layer 2 (30%)：Drama Amplifier（放大冲突、对话、张力的表面可见度）
- Layer 3 (20%)：Cost Visibility（让代价变得可见可感）
- Layer 4 (20%)：Decision Highlighting（让选择被读者看见）
- Layer 5 (10%)：Relationship Surface（让关系变化被表面化）

禁止：新增事件、新增角色、新增世界状态、修改决策结果
"""

import re
import json
import logging
from typing import Dict, Any, Optional

from openai import AsyncOpenAI

from src.agents.base import BaseAgent
from src.orchestrator.state import AgentState
from src.common.logging import setup_logging
from src.common.prompt_logger import log_prompt
from src.config import config
from src.config_loader import get_xianxia_config

logger = setup_logging("agents.rewrite")


class RewriteAgent(BaseAgent):
    """戏剧放大器 - 让读者感受到 Writer 已经写出来的困难"""

    # 是否启用 Drama Amplifier（可通过配置或环境变量控制）
    ENABLE_AMPLIFIER = getattr(config, 'experiment_enable_rewrite_v2', True)

    async def run(self, state: AgentState) -> Dict[str, Any]:
        raw = state.scene_text or ""
        if not raw or len(raw.strip()) < 50:
            return {"polished_text": raw, "scene_text": raw}

        # 提取戏剧结构（如果有）
        drama_struct = state.drama_structure or {}
        scene_role = state.scene_plan.get("scene_role") if state.scene_plan else None

        # ====== Layer 1: 基础清洁（始终执行） ======
        cleaned = self._basic_clean(raw)

        # ====== Layers 2-5: Drama Amplifier（如果启用且有戏剧结构） ======
        if self.ENABLE_AMPLIFIER and drama_struct and len(cleaned) > 200:
            enhanced = await self._amplify_drama(cleaned, drama_struct, scene_role)
            if enhanced:
                logger.info(f"RewriteAgent: drama amplification applied, {len(cleaned)} -> {len(enhanced)} chars")
                return {"scene_text": enhanced, "polished_text": enhanced}

        # 如果没有增强，返回基础清洁结果
        return {"scene_text": cleaned, "polished_text": cleaned}

    # ============================================================
    # Layer 1: 基础清洁（原有逻辑，略作整理）
    # ============================================================
    def _basic_clean(self, text: str) -> str:
        """基础清洁：删除禁止模式、合并短句、调整节奏、替换口头禅"""
        original = text

        # 1. 尝试解析 JSON（如果是 JSON 包裹的文本）
        is_json = False
        parsed = None
        try:
            parsed = json.loads(original)
            if isinstance(parsed, dict) and "scene_text" in parsed:
                is_json = True
                original = parsed.get("scene_text", "")
        except:
            pass

        if not original:
            return text

        config_obj = get_xianxia_config()
        voice = config_obj.voice

        # 1. 删除禁止模式
        forbidden = voice.get("repetition", {}).get("forbidden_patterns", [])
        for pattern in forbidden:
            original = re.sub(pattern, "", original)

        # 2. 替换高频口头禅（仅替换过度使用的）
        catchphrases = voice.get("dialogue", {}).get("catchphrases", [])
        for phrase in catchphrases:
            parts = original.split(phrase)
            if len(parts) > 2:   # 出现次数 > 1
                # 只保留第一次出现，后续替换为同义表达（简化：仅保留一次）
                original = parts[0] + phrase + "".join(parts[2:])

        # 3. 合并连续短句
        sentences = re.split(r'([。！？])', original)
        merged = []
        i = 0
        while i < len(sentences):
            if i + 1 < len(sentences):
                sent = sentences[i] + sentences[i+1]
            else:
                sent = sentences[i]
            # 如果当前句很短（<15字符）且不是对话开头，尝试合并下一句
            if len(sent.strip()) < 15 and not sent.strip().startswith('“'):
                if i + 2 < len(sentences):
                    sent += sentences[i+2] + (sentences[i+3] if i+3 < len(sentences) else "")
                    i += 2
            merged.append(sent)
            i += 2
        polished = "".join(merged)

        # 4. 调整连续短句（避免过多短句连续）
        lines = polished.split('\n')
        new_lines = []
        for line in lines:
            short_sentences = re.findall(r'[^。！？]+[。！？]', line)
            if len(short_sentences) > 3:
                # 将中间几个短句用逗号连接
                mid = "，".join(short_sentences[1:-1])
                line = short_sentences[0] + mid + short_sentences[-1]
            new_lines.append(line)
        polished = "\n".join(new_lines)

        # 5. 修复引号（将英文引号转为中文引号，但保留对话内容）
        polished = re.sub(r"'([^']*)'", r"“\1”", polished)

        # 6. 如果原输入是 JSON，重新包装
        if is_json:
            parsed["scene_text"] = polished
            return json.dumps(parsed, ensure_ascii=False)
        else:
            return polished

    # ============================================================
    # Layers 2-5: Drama Amplifier（LLM 驱动）
    # ============================================================
    async def _amplify_drama(self, text: str, drama_struct: Dict, scene_role: str = None) -> Optional[str]:
        """
        使用 LLM 进行戏剧放大，但不新增事件/角色/状态。
        返回增强后的文本，如果失败或越界则返回 None。
        """
        # 提取戏剧要素
        goal = drama_struct.get("scene_goal", "")
        obstacle = drama_struct.get("obstacle", {}).get("description", "")
        pressure = drama_struct.get("pressure", {}).get("description", "")
        decision = drama_struct.get("decision", {}).get("chosen", "")
        cost = drama_struct.get("cost", {}).get("success", "")
        rel_delta = drama_struct.get("relationship_delta", {})

        # 构建 Prompt
        prompt = self._build_amplify_prompt(text, goal, obstacle, pressure, decision, cost, rel_delta, scene_role)

        # 记录 Prompt 日志（用于调试）
        log_prompt("rewrite", prompt, metadata={"type": "drama_amplifier"})

        try:
            client = AsyncOpenAI(api_key="not-needed", base_url=config.llm_api_url)
            response = await client.chat.completions.create(
                model="Qwen3-32B-Q5_K_M-writer",   # 可配置
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,          # 较低温度，避免创造性偏离
                max_tokens=4096,
            )
            enhanced = response.choices[0].message.content or ""

            # 安全检查：不能太短（防止截断）
            if len(enhanced) < len(text) * 0.6:
                logger.warning(f"Enhanced text too short ({len(enhanced)} vs {len(text)}), falling back")
                return None

            # 安全检查：检测是否出现了新角色名（简单启发式）
            # 提取原文中的角色名（假设常见角色）
            import re
            original_chars = set(re.findall(r'[林逸|二叔|赵虎|李岩|苏清雪|玄老|管事|长老|执事|守卫|弟子|同门]', text))
            enhanced_chars = set(re.findall(r'[林逸|二叔|赵虎|李岩|苏清雪|玄老|管事|长老|执事|守卫|弟子|同门]', enhanced))
            # 如果增强文本引入了原文中没有的新角色名（排除极个别遗漏），则回退
            new_chars = enhanced_chars - original_chars
            # 如果新增了明显角色名（长度>=2），认为越界
            if any(len(c) >= 2 for c in new_chars):
                logger.warning(f"New characters detected in enhanced text: {new_chars}, falling back")
                return None

            # 检查是否新增了明显的事件关键词（如“忽然”、“突然”、“这时”加新动作）
            # 过于复杂，先不实现

            return enhanced
        except Exception as e:
            logger.error(f"Drama amplification failed: {e}")
            return None

    def _build_amplify_prompt(self, text: str, goal: str, obstacle: str, pressure: str,
                              decision: str, cost: str, rel_delta: Dict, scene_role: str = None) -> str:
        """构建戏剧放大 Prompt"""

        role_instructions = {
            "SETUP": "铺垫阶段：保持舒缓节奏，建立期待，埋下伏笔。",
            "ESCALATION": "冲突升级阶段：突出紧张感，加快节奏，强调压力和对抗。",
            "REVEAL": "揭示阶段：强化关键信息的冲击力，延长揭示瞬间的节奏。",
            "AFTERMATH": "余波阶段：展现后果，加深情感沉淀，让代价更可感。",
            "TRANSITION": "过渡阶段：自然衔接，节奏平稳。",
        }
        role_hint = role_instructions.get(scene_role, "保持场景的戏剧张力。")

        rel_text = ""
        if rel_delta.get("target") and rel_delta.get("to"):
            rel_text = f"关系变化：与 {rel_delta['target']} 的关系从 {rel_delta.get('from', '当前状态')} 变为 {rel_delta['to']}。"

        return f"""你是一位资深小说编辑，擅长**在不改变任何情节的前提下，让读者更强烈地感受到戏剧张力**。

## 核心原则（必须遵守）
1. **不新增事件** - 所有情节点保持原样，不添加新的发生事件
2. **不新增角色** - 不引入新人物
3. **不新增世界状态** - 不改变任何设定或背景
4. **不改变决策结果** - 角色的选择不能改
5. **只做放大** - 把已有的东西做得更明显

## 可用的放大手段
1. 在关键时刻增加感官细节（视觉、听觉、触觉、嗅觉）
2. 延长关键节奏（增加停顿、沉默、呼吸描写）
3. 让选择变得可见（在决策前插入权衡过程、内心犹豫）
4. 让代价变得可感（在代价发生后描述身体或情感感受）
5. 让关系变化表面化（在互动中增加关系信号，如语气、动作的细微变化）
6. 在对话中增加潜台词和情绪标签

## 本场景的戏剧结构（必须作为放大依据）
- 核心欲望：{goal}
- 阻碍：{obstacle}
- 压力：{pressure}
- 最终选择：{decision}
- 代价：{cost}
- {rel_text}

## 场景角色要求
{role_hint}

## 原文
{text}

## 输出要求
直接输出润色后的完整场景正文，不要添加任何分析、注释或说明。

**重要提醒**：不要添加任何新的事件、角色或设定。你的任务是让已有的内容更鲜明、更有感染力。"""

    # ============================================================
    # 可选：规则检查（防止越界）
    # ============================================================
    def _check_boundaries(self, original: str, enhanced: str) -> bool:
        """检查增强文本是否越界（新增事件/角色/状态）"""
        # 简单实现：检查是否出现原文中没有的新角色名
        # 使用正则提取所有中文姓名（2-4字）
        import re
        original_names = set(re.findall(r'[\u4e00-\u9fff]{2,4}', original))
        enhanced_names = set(re.findall(r'[\u4e00-\u9fff]{2,4}', enhanced))
        # 过滤掉常见非人名词
        stop_words = {'已经', '这个', '那个', '什么', '怎么', '可以', '因为', '所以', '但是', '就是', '如果', '然后', '时候', '知道', '觉得', '应该', '能够'}
        original_names -= stop_words
        enhanced_names -= stop_words
        new_names = enhanced_names - original_names
        if new_names:
            logger.warning(f"Potential new character names detected: {new_names}")
            return False
        return True