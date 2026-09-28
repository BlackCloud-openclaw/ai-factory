"""
Phase 15.3 — Shadow Rewrite Prompt Builder

构建用于 Shadow Rewrite 的 Prompt。
"""

import logging
from typing import Any, List

logger = logging.getLogger(__name__)


class ShadowPromptBuilder:
    """构建 Shadow Rewrite Prompt。"""

    def __init__(self, prompt_version: str = "phase15.3.v1"):
        self.prompt_version = prompt_version

    def build(self, text: str, contract: Any) -> str:
        """
        构建 Rewrite Prompt。
        """
        events = self._extract_events(contract)
        characters = self._extract_characters(contract)

        return f"""
下面是一段已经生成的小说正文。

【必须保留的事件】
{self._format_events(events)}

【必须保留的角色】
{characters}

【禁止新增】
- 不要增加新的剧情主线
- 不要增加新的世界事实（如"三年前曾来过这里"）
- 不要增加新角色
- 不要改变人物关系

你可以自由发挥的方面：
- 环境描写、感官细节
- 人物动作和身体反应
- 心理活动
- 对话之间的停顿、表情、动作
- 句子节奏和段落划分

你的任务不是重写故事，而是让读者"亲身经历"已经发生的这些事件。

直接输出改写后的正文，不要添加解释。

原文：
{text}
"""

    def _extract_events(self, contract: Any) -> List[str]:
        """从 contract 中提取必须事件。"""
        events = []
        try:
            if isinstance(contract, dict):
                observables = contract.get("observables", {})
                state_changes = observables.get("state_changes", [])
                for sc in state_changes:
                    sc_type = sc.get("type", "")
                    if sc_type == "plot_flag":
                        events.append(f"{sc.get('name')} 发生")
                    elif sc_type == "location_change":
                        events.append(f"{sc.get('actor')} 到达 {sc.get('location')}")
                    elif sc_type == "relationship_change":
                        events.append(f"{sc.get('from_char')} 与 {sc.get('to_char')} 关系变化 {sc.get('delta')}")
                    elif sc_type == "knowledge_gain":
                        events.append(f"获得知识：{sc.get('name')}")
                    elif sc_type == "inventory_acquire":
                        events.append(f"{sc.get('actor')} 获得 {sc.get('item')}")
            else:
                # 如果是 PlanningContract 对象
                if hasattr(contract, 'observables'):
                    for sc in contract.observables.state_changes:
                        sc_type = sc.type if hasattr(sc.type, 'value') else str(sc.type)
                        if sc_type == "plot_flag":
                            events.append(f"{sc.name} 发生")
                        elif sc_type == "location_change":
                            events.append(f"{sc.actor} 到达 {sc.location}")
                        elif sc_type == "relationship_change":
                            events.append(f"{sc.from_char} 与 {sc.to_char} 关系变化 {sc.delta}")
                        elif sc_type == "knowledge_gain":
                            events.append(f"获得知识：{sc.name}")
                        elif sc_type == "inventory_acquire":
                            events.append(f"{sc.actor} 获得 {sc.item}")
        except Exception as e:
            logger.warning(f"Failed to extract events from contract: {e}")

        return events if events else ["（从 contract 中未提取到具体事件）"]

    def _extract_characters(self, contract: Any) -> str:
        """从 contract 中提取角色。"""
        try:
            if isinstance(contract, dict):
                if "characters" in contract:
                    return ", ".join(contract["characters"])
                if "scene_plan" in contract:
                    return ", ".join(contract["scene_plan"].get("characters", []))
            else:
                if hasattr(contract, 'scene_plan') and contract.scene_plan:
                    return ", ".join(contract.scene_plan.get("characters", []))
        except Exception:
            pass
        return "（未提取到角色）"

    def _format_events(self, events: List[str]) -> str:
        if not events:
            return "（从 contract 中未提取到具体事件）"
        return "\n".join(f"- {e}" for e in events)