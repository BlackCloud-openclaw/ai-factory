# src/prompts/planner_prompts.py
import json
import re
import logging
import ast
from abc import ABC, abstractmethod
from typing import Dict, Any, Union, List
from src.orchestrator.state import AgentState
from src.domain.identity import get_main_character_name   # 新增导入

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
        protagonist = get_main_character_name()  # 动态获取
        return f"""你是一位专业小说策划师。请根据用户需求生成小说详细大纲，输出严格 JSON 格式，结构如下：
{{
    "title": "小说标题",
    "world_rules": ["规则1", "规则2"],
    "characters": [
        {{"name": "{protagonist}", "initial_state": {{"realm": "炼气", "level": 1}}}}
    ],
    "volumes": [
        {{
            "volume_num": 1,
            "title": "卷标题",
            "chapters": [
                {{
                    "chapter_num": 1,
                    "title": "章节标题",
                    "must_events": ["事件1", "事件2"],
                    "forbidden_events": []
                }}
            ]
        }}
    ]
}}

用户需求：{user_input}"""

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
        chapter_range = state.metadata.get("chapter_range", (1, chapters_per_vol))
        start_ch, end_ch = chapter_range
        num_to_gen = end_ch - start_ch + 1
        
        return f"""根据下面卷的信息，生成该卷的第 {start_ch} 到第 {end_ch} 章的详细大纲。
卷信息：卷{volume.get('volume_num')} 《{volume.get('title')}》
目标境界：{volume.get('target_realm')}
核心冲突：{volume.get('core_conflict')}

输出JSON数组，每个元素包含：
- chapter_num: 章节序号（必须是 {start_ch} 到 {end_ch} 之间的整数）
- title: 章节标题
- must_events: 必须发生的事件（字符串数组）
- forbidden_events: 禁止发生的事件（字符串数组）

输出格式：
[
    {{"chapter_num": {start_ch}, "title": "...", "must_events": [...], "forbidden_events": [...]}},
    ...
]

**重要**：
- 必须恰好生成 {num_to_gen} 个章节对象，序号连续。
- 不要输出任何额外文本，只输出 JSON 数组。
- 注意：如果前序章节已经发生过某些事件，请不要在本批次章节中重复（例如第1章“捡到玉佩”不要在第5章再出现）。"""

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

**🆕 每个场景必须包含 observables.state_changes**：描述场景结束后世界状态应发生的变化。至少一个，用于验证场景是否真正推进了故事状态。

**场景计划格式要求**：
每个场景是一个 JSON 对象，必须包含 "goal", "conflict", "outcome", "characters", "must_events", "observables", "scene_spec" 字段。

**剧情推进规则**：
- 不要重复已经在前几章完成的事件（例如第1章的“捡到神秘玉佩”不应在第2章及以后重复出现）。
- 剧情必须持续推进，不要停滞在同一个情节上。

**state_delta 示例**：
{{
    "character_updates": {{"林逸": {{"realm": "炼气三层", "hp": 95}}}},
    "relationship_updates": {{"林逸|二叔": -10}},
    "plot_flags": ["玉佩觉醒"]
}}

**depends_on 示例**：
- 场景2依赖场景1的结果：`"depends_on": [1]`
- 场景3无依赖：`"depends_on": []`

**observables.state_changes 详细说明**：
每个 state_change 包含：
- `type`: 类型（必填），可选值：`plot_flag`, `inventory_acquire`, `location_change`, `realm_change`, `relationship_change`, `knowledge_gain`
- `source`: 固定为 `"llm"`（表示由 Planner 直接生成）
- 根据类型不同，需要提供不同字段：
  - plot_flag: `"name"`（标记名）, `"value"`（true/false）
  - inventory_acquire: `"actor"`, `"item"`
  - location_change: `"actor"`, `"location"`
  - realm_change: `"actor"`, `"to_major_realm"`, `"to_minor_stage"`
  - relationship_change: `"from_char"`, `"to_char"`, `"delta"`（整数，正数表示友好）
  - knowledge_gain: `"name"`（知识名）, `"value"`（true）

**完整场景示例**（包含 observables）：
{{
    "goal": "林逸在丹室躲避追杀",
    "conflict": "炼丹长老步步紧逼",
    "outcome": "林逸触发禁制，长老受伤",
    "characters": ["林逸", "炼丹长老"],
    "must_events": ["触发丹室自爆禁制"],
    "observables": {{
        "state_changes": [
            {{ "type": "plot_flag", "name": "丹室自爆触发", "value": true, "source": "llm" }},
            {{ "type": "relationship_change", "from_char": "林逸", "to_char": "炼丹长老", "delta": -30, "source": "llm" }}
        ]
    }},
    "scene_spec": {{
        "world": {{ "location": "丹室", "time": "深夜", "atmosphere": "紧张", "sensory": ["火光", "爆炸声"] }},
        "reader_emotion": {{ "begin": "紧张", "middle": "震惊", "end": "释然" }},
        "narrative_function": "reveal_truth",
        "pov": "林逸"
    }},
    "depends_on": []
}}

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

        # ========== 熵控制动作注入（取代旧熵警告） ==========
        control_actions = state.metadata.get("entropy_control_actions", [])
        if control_actions:
            base_prompt += "\n\n【🔒 叙事稳态强制约束（优先级高于所有）】\n"
            for action in control_actions:
                action_type = action.get("type")
                params = action.get("params", {})
                if action_type == "limit_scene_role":
                    allowed = params.get("allowed", [])
                    forbidden = params.get("forbidden", [])
                    if allowed:
                        base_prompt += f"- 只允许使用以下场景角色: {', '.join(allowed)}。\n"
                    if forbidden:
                        base_prompt += f"- 禁止使用场景角色: {', '.join(forbidden)}。\n"
                elif action_type == "resolve_arcs":
                    max_open = params.get("max_open", 5)
                    arc_ids = params.get("arc_ids", [])
                    base_prompt += f"- 当前未解决弧线过多（超过 {max_open} 个）。"
                    if arc_ids:
                        base_prompt += f" 请优先解决以下弧线: {', '.join(arc_ids)}。\n"
                    else:
                        base_prompt += " 请优先安排弧线回收场景，不要新增弧线。\n"
                elif action_type == "forbid_new_lore":
                    duration = params.get("duration_chapters", 1)
                    lore_types = params.get("types", [])
                    if lore_types:
                        type_desc = []
                        if "character" in lore_types:
                            type_desc.append("新角色")
                        if "location" in lore_types:
                            type_desc.append("新地点")
                        if "item" in lore_types:
                            type_desc.append("新物品/道具")
                        if "realm" in lore_types:
                            type_desc.append("新境界/功法")
                        base_prompt += f"- 接下来 {duration} 章禁止引入: {', '.join(type_desc)}。\n"
                    else:
                        base_prompt += f"- 接下来 {duration} 章禁止引入新角色、新地点、新物品、新功法。\n"
                elif action_type == "force_low_stakes":
                    reason = params.get("reason", "熵过高")
                    base_prompt += f"- 强制进入低烈度章节（原因: {reason}），禁止战斗、突破、重大冲突。\n"
                elif action_type == "forbid_new_arcs":
                    duration = params.get("duration_chapters", 2)
                    base_prompt += f"- 接下来 {duration} 章禁止开启新的剧情弧线。\n"

        # 在 build 方法末尾，返回之前添加
        gravity_warning = state.metadata.get("gravity_warning")
        if gravity_warning:
            base_prompt += f"\n\n{gravity_warning}\n"

        # ========== v2.1: Scene Specification 生成指令 ==========
        spec_section = """
## 🆕 v2.1 场景规格（Scene Specification）

每个场景除 goal/conflict/outcome/must_events 外，必须包含 `scene_spec` 字段：

```json
{
    "scene_spec": {
        "world": {
            "location": "具体地点名",
            "time": "清晨|正午|黄昏|子夜|深夜",
            "atmosphere": "氛围关键词（潮湿/冷冽/宁静/肃杀/温暖/压抑）",
            "sensory": ["感官细节1", "感官细节2", "感官细节3"]
        },
        "reader_emotion": {
            "begin": "开头情绪",
            "middle": "中间情绪",
            "end": "结尾情绪"
        },
        "narrative_function": "introduce_mystery|escalate|reveal_truth|release_tension|transition|foreshadow",
        "pov": "视角角色名"
    }
}
字段说明
字段    说明    示例
world.location    场景发生地点    药园、议事堂、密室
world.time    场景时间    清晨、正午、黄昏、子夜
world.atmosphere    氛围基调    潮湿、冷冽、宁静、肃杀
world.sensory    感官细节（2-4个）    ["药香", "晨雾", "露水"]
reader_emotion.begin    开头读者情绪    好奇、警惕、平静
reader_emotion.middle    中间读者情绪    疑惑、愤怒、震惊
reader_emotion.end    结尾读者情绪    不安、冷静、沉重
narrative_function    场景叙事功能    introduce_mystery / escalate / reveal_truth / release_tension / transition
pov    视角角色    林逸
场景功能指南

    introduce_mystery：留下谜团，结尾产生悬念，不解释

    escalate：提升冲突，压力增大，局势紧张

    reveal_truth：揭示关键信息，让读者震惊或恍然大悟

    release_tension：缓解紧张，提供喘息空间

    transition：自然过渡，节奏平稳

情绪轨迹指南

    三者必须有变化（begin ≠ middle ≠ end）

    introduce_mystery：好奇 → 疑惑 → 不安

    escalate：警惕 → 愤怒 → 冷静

    reveal_truth：平静 → 震惊 → 沉重

    release_tension：疲惫 → 放松 → 坚定

    transition：平静 → 期待 → 平和

请为每个场景生成完整的 scene_spec。
"""

        return base_prompt + spec_section

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