"""
上下文编译器 - 将世界状态智能压缩为 LLM prompt

只注入当前场景必要的信息，避免 token 爆炸和上下文污染。
"""
import json
from .world_state import WorldState
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from .voiceprint import VoiceprintRegistry
from .state_projection import extract_hard_constraints, format_constraints_as_xml


class ContextCompiler:
    """上下文编译器"""
    
    def __init__(self, max_tokens: int = 3000):
        self.max_tokens = max_tokens
    
    def compile(
        self,
        world_state: WorldState,
        active_characters: Optional[List[str]] = None,
        max_active: int = 10,
        max_recent_events: int = 10,
        include_location: bool = True,
        include_global_flags: bool = True,
    ) -> str:
        """
        编译紧凑的上下文 JSON
        
        Args:
            world_state: 当前世界状态
            active_characters: 指定活跃角色列表（None 则自动选取）
            max_active: 最多包含多少个活跃角色
            max_recent_events: 最多包含多少个最近事件
            include_location: 是否包含当前位置信息
            include_global_flags: 是否包含全局标记
            
        Returns:
            JSON 字符串，可直接注入 prompt
        """
        # 确定活跃角色
        if active_characters is None:
            active_characters = world_state.get_active_characters(max_count=max_active)
        else:
            # 确保不超过限制
            active_characters = active_characters[:max_active]
        
        # 构建上下文
        context: Dict[str, Any] = {
            "active_characters": {},
            "recent_event_summaries": [],
        }
        
        # 添加角色状态（精简版）
        for name in active_characters:
            if name in world_state.characters:
                char = world_state.characters[name]
                context["active_characters"][name] = {
                    "realm": char.full_realm(),
                    "hp": char.hp,
                    "mp": char.mp,
                    "inventory": char.inventory[:5],  # 只显示前5个
                    "location": char.location,
                }
        
        # 添加关系网（只显示活跃角色之间的）
        context["relationships"] = {}
        for key, val in world_state.relationships.items():
            parts = key.split("|")
            if len(parts) == 2:
                a, b = parts
                if a in active_characters or b in active_characters:
                    context["relationships"][key] = val
        
        # 添加当前位置信息
        if include_location and world_state.map.current:
            current_loc = world_state.map.current
            context["current_location"] = current_loc
            if current_loc in world_state.map.locations:
                loc = world_state.map.locations[current_loc]
                context["location_info"] = {
                    "description": loc.description[:200] if loc.description else "",
                    "flags": loc.flags,
                }
        
        # 添加全局标记（只取 True 的布尔值和重要标记）
        if include_global_flags and world_state.global_flags:
            important_flags = {
                k: v for k, v in world_state.global_flags.items()
                if v is True or (isinstance(v, (int, float)) and v != 0)
            }
            if important_flags:
                context["global_flags"] = important_flags
        
        # 添加最近事件摘要（从 recent_event_ids 获取，需要外部提供摘要）
        # 这里只预留位置，摘要由调用方提供
        # 实际使用中，可以传入 recent_event_descriptions 参数
        
        # 序列化并限制长度
        result = json.dumps(context, ensure_ascii=False, indent=2)
        if len(result) > self.max_tokens:
            # 截断：移除关系网
            context.pop("relationships", None)
            result = json.dumps(context, ensure_ascii=False, indent=2)
            if len(result) > self.max_tokens:
                result = result[:self.max_tokens] + "\n... (truncated)"
        
        return result
    
    def compile_for_planner(
        self,
        world_state: WorldState,
        current_volume: int,
        current_chapter: int,
        outline: Dict[str, Any],
        max_tokens: int = 2000,
    ) -> str:
        """
        为 Planner 节点编译上下文
        """
        # 取前10个布尔标记作为列表
        recent_flags = [
            k for k, v in world_state.global_flags.items()
            if isinstance(v, bool) and v
        ][:10]
        
        context = {
            "current_position": {
                "volume": current_volume,
                "chapter": current_chapter,
            },
            "story_state": {
                "active_characters_count": len(world_state.characters),
                "main_character": list(world_state.characters.keys())[:3] if world_state.characters else [],
                "current_location": world_state.map.current,
            },
            "outline_summary": {
                "title": outline.get("title", ""),
                "total_volumes": len(outline.get("volumes", [])),
                "current_volume_info": None,
            },
            "recent_flags": recent_flags,
        }
        
        # 添加当前卷信息
        volumes = outline.get("volumes", [])
        if 0 <= current_volume - 1 < len(volumes):
            context["outline_summary"]["current_volume_info"] = volumes[current_volume - 1]
        
        result = json.dumps(context, ensure_ascii=False, indent=2)
        if len(result) > max_tokens:
            result = result[:max_tokens] + "\n... (truncated)"
        return result

    def build_writer_prompt(self, scene_plan, world_state, voiceprint_registry, compiled_context, history_summaries: List[str] = None, key_events_summary: str = None, voice_memory=None):
        lines = []
        
        # ========== 硬状态注入（最高优先级）==========
        constraints = extract_hard_constraints(world_state)
        
        # 添加历史摘要（如果提供）
        if history_summaries:
            lines.append("【历史章节摘要】")
            for s in history_summaries:
                lines.append(s)
            lines.append("")
        
        if key_events_summary:
            lines.append(key_events_summary)
            lines.append("")        
            
        # 添加醒目的境界约束 banner
        lines.append("=" * 60)
        lines.append("⚠️ 当前境界约束 - 必须严格遵守 ⚠️")
        lines.append("=" * 60)
        current_realm = constraints.get("current_realm", {})
        lines.append(f"当前境界：{current_realm.get('full', '未知')}")
        allowed = constraints.get("allowed_next_realms", [])
        if allowed:
            lines.append(f"允许突破：{', '.join(allowed)}")
        else:
            lines.append("允许突破：当前无法突破")
        forbidden = constraints.get("forbidden_realms", [])
        if forbidden:
            lines.append(f"禁止突破：{', '.join(forbidden[:5])}{'...' if len(forbidden) > 5 else ''}")
        lines.append("")
        lines.append("【重要】每次突破只能提升一个小境界，禁止跨越大境界！")
        lines.append("")
        
        # 原有的 XML 格式约束
        lines.append(format_constraints_as_xml(constraints))
        lines.append("")

        # 1. 注入编译后的世界状态
        lines.append("【当前世界状态摘要】")
        lines.append(compiled_context)
        lines.append("")
        
        # 2. 场景计划（强制要求）
        goal = scene_plan.get("goal", "")
        conflict = scene_plan.get("conflict", "")
        outcome = scene_plan.get("outcome", "")
        characters = scene_plan.get("characters", [])
        must_events = scene_plan.get("must_events", [])
        
        lines.append("【🔴 强制要求：你必须严格遵循以下计划】")
        if goal:
            lines.append(f"🎯 场景目标：{goal}")
            lines.append("   → 整段正文必须围绕实现此目标展开。")
        if conflict:
            lines.append(f"⚔️ 核心冲突：{conflict}")
            lines.append("   → 必须在正文中明确体现此冲突。")
        if outcome:
            lines.append(f"🏁 预期结果：{outcome}")
        if characters:
            lines.append(f"👥 参与角色：{', '.join(characters)}")
        lines.append("")
        
        # 3. 必须事件（强制输出精确短语）
        if must_events:
            lines.append("【🔴 必须发生的事件（缺一不可）】")
            lines.append("你必须在 scene_text 中**原样包含**以下短语（不能改写）：")
            for i, evt in enumerate(must_events, 1):
                lines.append(f"{i}. 「{evt}」")
            lines.append("")
            lines.append("⚠️ 注意：上述事件必须**逐字逐句**出现在正文中，不可省略或改写！")
            lines.append("   例如：「拜入青云宗外门」必须原样写出，不能写成「成为外门弟子」。")
            lines.append("")
        
        # 4. 角色语言约束（从 Voiceprint 获取）
        if characters:
            lines.append("【角色语言风格约束】")
            for char in characters:
                constraint = voiceprint_registry.build_prompt_constraint(char)
                if constraint:
                    lines.append(constraint)
            lines.append("")        
        
        # 5. 结构化输出要求（增强 events 字段指导）
        lines.append("【输出格式要求】")
        lines.append("你必须严格按照以下 JSON 格式输出，不要添加任何额外解释：")
        lines.append('{')
        lines.append('    "scene_text": "场景正文（纯文本，不要包含 JSON）",')
        lines.append('    "events": [  // 根据场景中发生的实际变化，选择合适的类型，不要总是写 realm_upgrade',)
        lines.append('        {"type": "item_acquire", "actor": "林逸", "item": "神秘玉佩", "source": "捡到"},')
        lines.append('        {"type": "relationship_change", "from_char": "林逸", "to_char": "二叔", "delta": -10},')
        lines.append('        {"type": "location_enter", "actor": "林逸", "location": "禁地"},')
        lines.append('        {"type": "plot_flag_set", "flag": "玉佩觉醒", "value": true}')
        lines.append('    ],')
        lines.append('    "foreshadowing": ["后续可发展的伏笔1", "伏笔2"]')
        lines.append('}')
        lines.append("")
        
        # 5.1 JSON 格式正确性强调（新增）
        lines.append("⚠️ 特别注意 JSON 格式的正确性：")
        lines.append("1. 字符串内的双引号必须转义，例如 \\\" 表示一个双引号字符。")
        lines.append("2. 字符串内不得包含未转义的控制字符（如换行符、制表符），请使用 \\n、\\t 转义。")
        lines.append("3. 不要在键或字符串值末尾留下多余逗号。")
        lines.append("4. 确保所有花括号、方括号正确闭合。")
        lines.append("")
        
        # 添加事件类型说明（仅作参考，不强制）
        lines.append("【events 字段填写指南】")
        lines.append("根据场景中实际发生的状态变化，从以下类型中选择，可以写多个事件：")
        lines.append('- realm_upgrade: 境界突破（actor, to_major_realm: "金丹", to_minor_stage: 1）')
        lines.append("- item_acquire: 获得物品（actor, item, quantity, source可选）")
        lines.append("- item_lose: 失去物品（actor, item, quantity, reason）")
        lines.append("- relationship_change: 关系变化（from_char, to_char, delta）")
        lines.append("- location_enter: 进入新地点（actor, location）")
        lines.append("- plot_flag_set: 设置剧情标记（flag, value）")
        lines.append("- hp_changed / mp_changed: 血量/灵力变化（actor, delta, new_hp）")
        lines.append("- combat_result: 战斗结果（winner, loser, result）")
        lines.append("- dialogue: 关键对话（speaker, listener, summary）")
        lines.append("- discovery: 发现重要线索（discoverer, discovery, importance）")
        lines.append("- npc_introduce: 新角色登场（name, role）")
        lines.append("")
        lines.append("⚠️ 注意：")
        lines.append("- 每个场景至少应包含1-2个事件，除非是纯过渡场景。")
        lines.append("- 不要虚构不存在的事件类型，只使用上述列表中的类型。")
        lines.append("- 如果没有任何状态变化，events 可以为空数组 []。")
        lines.append("")
        
        # 6. 写作规则（强调）
        lines.append("【写作规则】")
        lines.append("1. 必须**原样包含**上述【必须发生的事件】中的所有短语")
        lines.append("2. 必须实现【强制要求】中的场景目标和冲突")
        lines.append("3. 每个必须事件至少用 100 字展开描写")
        lines.append("4. 对话必须符合【角色语言风格约束】")
        lines.append("5. 严禁输出任何思考过程、分析、计划、括号注释")
        lines.append("6. 直接输出 JSON，scene_text 字段内是小说正文")
        lines.append("7. 不要重复已经完成的事件")
        lines.append("8. 如果某个必须事件在之前章节已经发生过，在本章节中不要再次出现")
        lines.append("9. **events 数组请根据实际剧情填写，不要总是写 realm_upgrade**")
        lines.append("10. 输出前请检查 JSON 格式是否正确，确保没有语法错误。")

        lines.append("- 如果引入了全新设定（新角色/新地点/新物品/新功法），请在事件中添加 'new_lore': true")

        if voice_memory:
            style_constraints = voice_memory.get_style_constraints_prompt()
            if style_constraints:
                lines.append("\n" + style_constraints)

        return "\n".join(lines)