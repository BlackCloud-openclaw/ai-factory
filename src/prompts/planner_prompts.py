# src/prompts/planner_prompts.py
import json
import re
from abc import ABC, abstractmethod
from typing import Dict, Any
from src.orchestrator.state import AgentState

def extract_json_from_response(text: str) -> Dict[str, Any]:
    """从 LLM 响应中提取第一个完整的 JSON 对象或数组"""
    text = text.strip()
    # 尝试提取 markdown 代码块中的 JSON
    match = re.search(r'```(?:json)?\s*\n([\s\S]*?)\n\s*```', text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    # 找到第一个 { 或 [ 并匹配闭合
    start = None
    for i, ch in enumerate(text):
        if ch in '{[':
            start = i
            break
    if start is None:
        raise ValueError("No JSON object or array found")
    stack = []
    end = start
    for i in range(start, len(text)):
        ch = text[i]
        if ch in '{[':
            stack.append(ch)
        elif ch == '}':
            if stack and stack[-1] == '{':
                stack.pop()
            else:
                break
        elif ch == ']':
            if stack and stack[-1] == '[':
                stack.pop()
            else:
                break
        if not stack:
            end = i
            break
    json_str = text[start:end+1]
    # 尝试解析
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # 尝试修复常见问题：尾随逗号、单引号等（简单处理）
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        json_str = re.sub(r"'", '"', json_str)  # 将单引号改为双引号（可能破坏内容）
        return json.loads(json_str)

class PromptBuilder(ABC):
    @abstractmethod
    def build(self, state: AgentState) -> str:
        pass

    @abstractmethod
    def parse_response(self, response: str) -> Dict[str, Any]:
        pass

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


class ScenePlanPromptBuilder(PromptBuilder):
    def build(self, state: AgentState) -> str:
        outline = state.outline
        chapter = state.current_chapter
        return f"""根据以下大纲和当前章节号，生成该章节的 3~5 个场景计划（JSON 数组）。

大纲：{outline}
当前章号：{chapter}

每个场景计划是一个 JSON 对象，包含以下字段：
{{
    "goal": "场景目标",
    "conflict": "冲突描述",
    "outcome": "结果",
    "characters": ["角色1", "角色2"]
}}

输出格式：一个 JSON 数组，例如 [{{...}}, {{...}}]。不要输出任何额外文本。"""

    def parse_response(self, response: str) -> Dict[str, Any]:
        data = extract_json_from_response(response)   # 复用已有的 JSON 提取函数
        if isinstance(data, list):
            return {"scenes": data}
        elif isinstance(data, dict) and "scenes" in data:
            return data
        else:
            # 降级：包装成单场景列表
            return {"scenes": [data] if isinstance(data, dict) else []}


PROMPT_REGISTRY = {
    "code": CodePromptBuilder(),
    "novel_outline": NovelOutlinePromptBuilder(),
    "scene_plan": ScenePlanPromptBuilder(),
}