# src/prompts/planner_prompts.py
import json
import re
import logging
import ast
from abc import ABC, abstractmethod
from typing import Dict, Any, Union, List
from src.orchestrator.state import AgentState

# 尝试导入 json_repair，如果不存在则降级
try:
    from json_repair import repair_json
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False
    repair_json = None

logger = logging.getLogger("agents.planner")

def extract_json_from_response(text: str) -> Union[Dict[str, Any], List[Any]]:
    """
    从 LLM 响应中提取 JSON 对象或数组，具有极强的容错性。
    优先使用 json_repair 库，然后回退到手动修复。
    """
    text = text.strip()
    logger.debug(f"Extracting JSON from response (length={len(text)})")

    # ========== 1. 提取 markdown 代码块 ==========
    match = re.search(r'```(?:json)?\s*\n([\s\S]*?)\n\s*```', text, re.DOTALL)
    if match:
        text = match.group(1).strip()
        logger.debug("Extracted content from markdown code block")

    # ========== 2. 尝试使用 json_repair 库 ==========
    if HAS_JSON_REPAIR:
        try:
            # json_repair 会尝试修复常见错误（缺失引号、尾随逗号、注释等）
            repaired = repair_json(text)
            parsed = json.loads(repaired)
            logger.info("Successfully parsed JSON using json_repair")
            return parsed
        except Exception as e:
            logger.warning(f"json_repair failed: {e}, falling back to manual repair")
    else:
        logger.debug("json_repair not installed, using manual repair")

    # ========== 3. 手动修复（原有逻辑） ==========
    # 查找第一个数组 '[' 或对象 '{'
    start_idx = text.find('[')
    if start_idx == -1:
        start_idx = text.find('{')
        if start_idx == -1:
            raise ValueError("No JSON array or object found")

    # 匹配完整的 JSON 结构
    stack = []
    end = start_idx
    for i, ch in enumerate(text[start_idx:], start_idx):
        if ch in '{[':
            stack.append(ch)
        elif ch == '}':
            if stack and stack[-1] == '{':
                stack.pop()
            else:
                if len(stack) == 0:
                    end = i
                    break
                continue
        elif ch == ']':
            if stack and stack[-1] == '[':
                stack.pop()
            else:
                if len(stack) == 0:
                    end = i
                    break
                continue
        if not stack:
            end = i
            break
    else:
        end = len(text) - 1

    json_str = text[start_idx:end+1]

    # 去除尾随逗号
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)

    # 修复字符串值内部未转义的双引号
    def fix_quotes(m):
        key = m.group(1)
        value = m.group(2)
        value_fixed = re.sub(r'(?<!\\)"', r'\\"', value)
        return f'"{key}": "{value_fixed}"'

    for _ in range(3):
        new_json = re.sub(r'"([^"\\]+)"\s*:\s*"([^"]*)"', fix_quotes, json_str)
        if new_json == json_str:
            break
        json_str = new_json

    # 转义控制字符
    def escape_control(m):
        ch = m.group(0)
        if ch in '\t\n\r':
            return ch
        return f'\\u{ord(ch):04x}'

    json_str = json_str.replace('\\\\', '<<DBACK>>')
    json_str = re.sub(r'[\x00-\x1F]', escape_control, json_str)
    json_str = json_str.replace('<<DBACK>>', '\\\\')

    # 尝试解析整个 JSON
    try:
        return json.loads(json_str)
    except Exception as e:
        logger.warning(f"Whole JSON parse failed after fixes: {e}")

    # 最后手段：按对象分割解析（适用于数组）
    if json_str.startswith('[') and json_str.endswith(']'):
        content = json_str[1:-1].strip()
        objects = []
        brace_count = 0
        start_obj = 0
        in_string = False
        for i, ch in enumerate(content):
            if ch == '"' and (i == 0 or content[i-1] != '\\'):
                in_string = not in_string
            if not in_string:
                if ch == '{':
                    if brace_count == 0:
                        start_obj = i
                    brace_count += 1
                elif ch == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        obj_str = content[start_obj:i+1]
                        try:
                            obj_str = re.sub(r',\s*}', '}', obj_str)
                            obj = json.loads(obj_str)
                            objects.append(obj)
                        except:
                            try:
                                obj = ast.literal_eval(obj_str)
                                objects.append(obj)
                            except:
                                logger.warning(f"Skipping invalid object: {obj_str[:100]}")
        if objects:
            return objects
        else:
            raise ValueError("No valid objects found after split parsing")
    else:
        raise ValueError("Not a JSON array")
    

class PromptBuilder(ABC):
    @abstractmethod
    def build(self, state: AgentState) -> str:
        pass

    @abstractmethod
    def parse_response(self, response: str) -> Dict[str, Any]:
        pass


# ---------- 代码生成任务 ----------
class CodePromptBuilder(PromptBuilder):
    def build(self, state: AgentState) -> str:
        user_request = state.user_input
        return f"""You are a Task Planner. Break down the user request into a sequence of subtasks (each with a type, description, and dependencies). Use only these types: code, research, validate.

Return ONLY a valid JSON object with this structure:
{{
    "plan_id": "unique_id",
    "subtasks": [
        {{
            "id": "task_1",
            "name": "short name",
            "description": "detailed instruction",
            "type": "code",
            "dependencies": []
        }}
    ]
}}

User request: {user_request}"""

    def parse_response(self, response: str) -> Dict[str, Any]:
        return extract_json_from_response(response)


# ---------- 一次性大纲生成器（保留兼容） ----------
class NovelOutlinePromptBuilder(PromptBuilder):
    def build(self, state: AgentState) -> str:
        user_input = state.user_input
        return f"""You are a professional novel planner. Generate a detailed outline for a novel based on the user's requirements.

Output strict JSON with the following structure:
{{
    "title": "novel title",
    "world_rules": ["rule1", "rule2"],
    "characters": [
        {{"name": "林风", "initial_state": {{"realm": "炼气", "level": 1}}}}
    ],
    "volumes": [
        {{
            "volume_num": 1,
            "title": "volume title",
            "chapters": [
                {{
                    "chapter_num": 1,
                    "title": "chapter title",
                    "must_events": ["event1", "event2"],
                    "forbidden_events": []
                }}
            ]
        }}
    ]
}}

User request: {user_input}"""

    def parse_response(self, response: str) -> Dict[str, Any]:
        return extract_json_from_response(response)


# ---------- 分步式大纲生成器 ----------
class OutlineVolumesPromptBuilder(PromptBuilder):
    def build(self, state: AgentState) -> str:
        user_input = state.user_input
        return f"""你是一位小说策划专家。请根据用户需求，生成小说卷级别的概览（JSON数组）。
每个元素包含：
- volume_num: 卷序号（从1开始）
- title: 卷标题（简洁有力）
- target_realm: 主角在本卷结束时应该达到的境界
- core_conflict: 本卷的核心冲突（一句话）

用户需求：{user_input}

输出格式严格为JSON数组，例如：
[
    {{"volume_num": 1, "title": "初入修仙界", "target_realm": "筑基", "core_conflict": "入门之争与机缘争夺"}},
    {{"volume_num": 2, "title": "宗门试炼", "target_realm": "金丹", "core_conflict": "宗门大比与内奸阴谋"}}
]
请确保总卷数符合用户要求。"""

    def parse_response(self, response: str) -> Dict[str, Any]:
        data = extract_json_from_response(response)
        if isinstance(data, list):
            return {"volumes": data}
        return {"volumes": []}


class OutlineChaptersPromptBuilder(PromptBuilder):
    def build(self, state: AgentState) -> str:
        volume = state.metadata.get("current_volume_info", {})
        chapters_per_vol = state.metadata.get("chapters_per_vol", 10)
        return f"""根据下面卷的信息，生成该卷的 {chapters_per_vol} 章详细大纲。
卷信息：卷{volume.get('volume_num')} 《{volume.get('title')}》
目标境界：{volume.get('target_realm')}
核心冲突：{volume.get('core_conflict')}

输出JSON数组，每个元素包含：
- chapter_num: 章节序号
- title: 章节标题
- must_events: 必须发生的事件（字符串数组）
- forbidden_events: 禁止发生的事件（字符串数组）

输出格式：
[
    {{"chapter_num": 1, "title": "初遇机缘", "must_events": ["捡到神秘玉佩"], "forbidden_events": []}},
    ...
]
注意：长度必须恰好为 {chapters_per_vol} 章。"""

    def parse_response(self, response: str) -> Dict[str, Any]:
        data = extract_json_from_response(response)
        if isinstance(data, list):
            return {"chapters": data}
        return {"chapters": []}


# ---------- 场景计划生成器（新架构增强版） ----------
class ScenePlanPromptBuilder(PromptBuilder):
    def build(self, state: AgentState) -> str:
        outline = state.outline
        chapter = state.current_chapter
        volume = getattr(state, 'current_volume', 1)
        
        # 添加调试日志
        import logging
        logger = logging.getLogger("agents.planner")
        logger.info(f"ScenePlanPrompt: volume={volume}, chapter={chapter}, outline exists: {outline is not None}")
        
        # 提取当前章节的 must_events
        current_must_events = []
        current_chapter_title = ""
        if outline and isinstance(outline, dict) and "volumes" in outline:
            volumes = outline.get("volumes", [])
            vol_idx = volume - 1
            if 0 <= vol_idx < len(volumes):
                chapters = volumes[vol_idx].get("chapters", [])
                ch_idx = chapter - 1
                if 0 <= ch_idx < len(chapters):
                    current_must_events = chapters[ch_idx].get("must_events", [])
                    current_chapter_title = chapters[ch_idx].get("title", "")
                    logger.info(f"Found must_events for chapter {chapter}: {current_must_events}")
                else:
                    logger.warning(f"Chapter index {ch_idx} out of range (chapters len={len(chapters)})")
            else:
                logger.warning(f"Volume index {vol_idx} out of range (volumes len={len(volumes)})")
        else:
            logger.warning(f"Outline structure invalid: {type(outline)}")
        
        must_events_text = "\n".join(f"- {e}" for e in current_must_events) if current_must_events else "（本章大纲未定义必须事件，请根据前情合理推进剧情）"
        
        base_prompt = f"""根据以下大纲和当前章节号，生成该章节的 3~5 个场景计划（JSON 数组）。

大纲：{outline}
当前卷：{volume}
当前章号：{chapter}
当前章标题：{current_chapter_title}

**⚠️ 本章必须包含的剧情事件（必须覆盖以下所有 must_events）**：
{must_events_text}

**must_events 分配规则**：
- 将上述必须事件**分配到不同场景**中，每个场景的 "must_events" 只能是这些事件的子集。
- 所有场景的 must_events **合集必须完整覆盖**全部必须事件，且各场景之间**不能重复**相同事件。
- 如果某场景不包含任何必须事件，`"must_events": []` 是允许的（例如示例中的场景3）。

**场景计划格式要求**：
每个场景是一个 JSON 对象，必须包含 "goal", "conflict", "outcome", "characters", "must_events" 字段。

**剧情推进规则**：
- 不要重复已经在前几章完成的事件（例如第1章的“捡到神秘玉佩”不应在第2章及以后重复出现）。
- 剧情必须持续推进，不要停滞在同一个情节上。

**示例**：
{{
    "goal": "场景目标",
    "conflict": "冲突描述",
    "outcome": "结果",
    "characters": ["角色1", "角色2"],
    "must_events": ["该场景需要体现的其中一个必须事件"]
}}

**state_delta 示例**：
{{
    "character_updates": {{"林逸": {{"realm": "炼气三层", "hp": 95}}}},
    "relationship_updates": {{"林逸|二叔": -10}},
    "plot_flags": ["玉佩觉醒"]
}}

**depends_on 示例**：
- 场景2依赖场景1的结果：`"depends_on": [1]`
- 场景3无依赖：`"depends_on": []`

输出格式：一个 JSON 数组，每个元素包含上述所有字段。不要输出任何额外文本。"""

        # 注入压缩上下文（可选）
        compressed_context = state.metadata.get("compressed_context")
        if compressed_context:
            base_prompt += f"\n\n【参考历史剧情摘要】\n{compressed_context}\n请根据以上历史信息，规划接下来的场景，保持剧情连贯。"

        # 注入历史章节摘要（可选）
        history_summaries = state.metadata.get("history_summaries")
        if history_summaries:
            summary_text = "\n\n【参考历史章节摘要】\n" + "\n---\n".join(history_summaries)
            base_prompt += summary_text

        affordance_hints = state.metadata.get("affordance_hints", [])
        if affordance_hints:
            base_prompt += "\n\n【当前世界允许的能力】\n"
            for hint in affordance_hints:
                base_prompt += f"- {hint}\n"
            base_prompt += "建议：生成的事件最好不超出上述能力范围，但允许特殊情况。\n"

        return base_prompt

    def parse_response(self, response: str) -> Dict[str, Any]:
        data = extract_json_from_response(response)
        if isinstance(data, list):
            return {"scenes": data}
        elif isinstance(data, dict) and "scenes" in data:
            return data
        else:
            return {"scenes": [data] if isinstance(data, dict) else []}
        
# ---------- 注册表 ----------
PROMPT_REGISTRY = {
    "code": CodePromptBuilder(),
    "novel_outline": NovelOutlinePromptBuilder(),
    "volumes_outline": OutlineVolumesPromptBuilder(),
    "chapters_outline": OutlineChaptersPromptBuilder(),
    "scene_plan": ScenePlanPromptBuilder(),
}