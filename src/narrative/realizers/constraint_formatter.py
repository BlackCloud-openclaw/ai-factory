# src/narrative/realizers/constraint_formatter.py

"""
Constraint 格式化器 — 将 opaque payload 转换为 LLM 可读的约束摘要

注意：这个模块是 Runtime Adapter 的职责，不是 Narrative 协议的职责。
ReferenceRealizer 接收的是已经格式化好的字符串，而不是解析 payload。
"""

from typing import Any, Dict


def format_constraint_payload(payload: Any) -> str:
    """
    将 Constraint payload 转换为人类可读的摘要

    这是 Runtime Adapter 应该提供的功能。
    这里提供默认实现，但调用方应优先使用 Runtime 提供的格式化器。
    """
    if not payload:
        return "- 无特殊约束（仅保持剧情事实一致）"

    if isinstance(payload, dict):
        lines = []
        if "events" in payload:
            events = payload["events"]
            if isinstance(events, list):
                lines.append(f"- 事件顺序：{', '.join(str(e)[:30] for e in events[:5])}")
        if "plot_flags" in payload:
            flags = payload["plot_flags"]
            if isinstance(flags, dict):
                lines.append(f"- 剧情标记：{', '.join(list(flags.keys())[:5])}")
        if "character_states" in payload:
            chars = payload["character_states"]
            if isinstance(chars, dict):
                lines.append(f"- 角色状态：{', '.join(list(chars.keys())[:3])}")
        if "timeline" in payload:
            timeline = payload["timeline"]
            if isinstance(timeline, list):
                lines.append(f"- 时间线：{', '.join(str(e)[:20] for e in timeline[:5])}")
        return "\n".join(lines) if lines else "- 无特殊约束"

    return "- 无特殊约束"