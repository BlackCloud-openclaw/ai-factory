from .models import JudgeDimension

# Prompt 版本（单一来源）
PROMPT_VERSIONS = {
    JudgeDimension.CONTINUITY: "1.0",
    JudgeDimension.CHARACTER: "1.0",
    JudgeDimension.DIALOGUE: "1.0",
    JudgeDimension.FLOW: "1.0",
}

CONTINUITY_JUDGE_PROMPT = """
你是一位专业的小说编辑。评估以下两个连续场景之间的叙事连续性。

**场景 A（上一场景结尾）**：
{scene_before}

**场景 B（当前场景开头）**：
{scene_after}

请评估以下维度（0.0-1.0 分）：
1. 事件衔接：当前场景是否自然承接了上一场景？
2. 时间一致性：时间流动是否合理？
3. 空间一致性：地点变化是否有明确交代？
4. 角色状态：角色情绪、位置、状态是否连续？

输出 JSON（不要添加任何额外文本）：
{
    "continuity_score": 0.0-1.0,
    "reasoning": "...",
    "details": {
        "event_continuity": 0.0-1.0,
        "time_consistency": 0.0-1.0,
        "space_consistency": 0.0-1.0,
        "character_state": 0.0-1.0
    }
}
"""

CHARACTER_JUDGE_PROMPT = """
你是一位专业的小说编辑。评估当前场景中的角色行为一致性。

**场景文本**：
{scene_text}

**角色信息**：
{character_info}

请评估以下维度（0.0-1.0 分）：
1. 行为一致性：角色的行为是否符合其历史设定？
2. 语言一致性：角色的对话是否符合其语言风格？
3. 动机合理性：角色的行动是否有合理的动机支撑？
4. 关系动态：角色间的关系是否符合预期？

输出 JSON（不要添加任何额外文本）：
{
    "character_score": 0.0-1.0,
    "reasoning": "...",
    "details": {
        "behavior": 0.0-1.0,
        "language": 0.0-1.0,
        "motivation": 0.0-1.0,
        "relationship": 0.0-1.0
    }
}
"""

DIALOGUE_JUDGE_PROMPT = """
你是一位专业的小说编辑。评估当前场景中的对话质量。

**场景文本**：
{scene_text}

请评估以下维度（0.0-1.0 分）：
1. 自然度：对话是否自然、真实？
2. 信息量：对话是否推动了情节或揭示了信息？
3. 潜台词：对话中是否有隐含的意图或冲突？
4. 角色区分度：不同角色的对话风格是否可区分？

输出 JSON（不要添加任何额外文本）：
{
    "dialogue_score": 0.0-1.0,
    "reasoning": "...",
    "details": {
        "naturalness": 0.0-1.0,
        "information": 0.0-1.0,
        "subtext": 0.0-1.0,
        "distinctiveness": 0.0-1.0
    }
}
"""

FLOW_JUDGE_PROMPT = """
你是一位专业的小说编辑。评估当前场景的阅读流畅度。

**场景文本**：
{scene_text}

请评估以下维度（0.0-1.0 分）：
1. 节奏感：场景节奏是否恰当？
2. 信息密度：信息分布是否合理？
3. 情感张力：情感曲线是否有起伏？
4. 结构完整性：场景是否有清晰的开始、发展和结束？

输出 JSON（不要添加任何额外文本）：
{
    "flow_score": 0.0-1.0,
    "reasoning": "...",
    "details": {
        "pacing": 0.0-1.0,
        "information_density": 0.0-1.0,
        "emotional_tension": 0.0-1.0,
        "structure": 0.0-1.0
    }
}
"""