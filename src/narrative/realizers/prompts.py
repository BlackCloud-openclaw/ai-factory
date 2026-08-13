# src/narrative/realizers/prompts.py

from typing import List, Optional

from src.narrative.intent import NarrativeIntent, IntentPriority
from src.narrative.context import NarrativeContext


def build_editor_prompt(
    artifact_text: str,
    context: NarrativeContext,
    intents: List[NarrativeIntent],
    constraint_summary: str,
    resolution_text: Optional[str] = None,
) -> str:
    lines = []

    # 1. 角色设定
    lines.append("你是一位资深小说责任编辑。")
    lines.append("你的任务是修改以下章节，在不改变任何剧情事实的前提下，提升阅读体验。")
    lines.append("")

    # 2. 当前文本
    lines.append("【原文】")
    lines.append("```text")
    lines.append(artifact_text)
    lines.append("```")
    lines.append("")

    # 3. 上下文信息
    lines.append("【上下文】")
    if context.metadata:
        lines.append(f"- 第 {context.metadata.volume} 卷第 {context.metadata.chapter} 章")
        lines.append(f"- 场景 {context.metadata.scene_index + 1}/{context.metadata.total_scenes}")
    lines.append("")

    # 4. 决议部分（如果存在）
    if resolution_text:
        lines.append("【冲突解决决策】")
        lines.append(resolution_text)
        lines.append("")

    # 5. 编辑意图（按优先级排序）
    lines.append("【编辑目标】")
    priority_order = {
        IntentPriority.HIGH: 0,
        IntentPriority.MEDIUM: 1,
        IntentPriority.LOW: 2,
    }
    sorted_intents = sorted(
        intents,
        key=lambda i: priority_order.get(i.priority, 1),
    )

    for i, intent in enumerate(sorted_intents, 1):
        source = intent.source.value
        effect = intent.desired_effect
        lines.append(f"{i}. [{source}] {effect}")
        if intent.preserve:
            lines.append(f"   保留：{', '.join(intent.preserve)}")
        if intent.avoid:
            lines.append(f"   避免：{', '.join(intent.avoid)}")
        if intent.rationale:
            lines.append(f"   原因：{intent.rationale}")
    lines.append("")

    # 6. 约束（不可修改）
    lines.append("【不可修改的内容】")
    lines.append(constraint_summary)
    lines.append("")

    # 7. 输出要求
    lines.append("【输出要求】")
    lines.append("1. 只输出修改后的完整章节，不要添加任何解释或注释")
    lines.append("2. 不要改变任何剧情事件、人物状态、时间线")
    lines.append("3. 可以增加：环境描写、心理描写、过渡段落、对话优化")
    lines.append("4. 保持原文的叙事风格和基本结构")
    lines.append("5. 优先解决 High 优先级的问题")
    lines.append("6. 确保最终结果与原文长度相近（±20%）")
    lines.append("")
    lines.append("请直接输出修改后的完整章节：")

    return "\n".join(lines)