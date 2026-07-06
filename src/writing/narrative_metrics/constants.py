"""
Narrative KPI Constants - Phase 5
冻结的 Spec 定义，所有评分逻辑以此为准。
"""

from typing import Dict, List, Tuple

# 维度 ID
DIMENSIONS = [
    "dialogue",      # Dialogue Richness
    "interaction",   # Interaction
    "conflict",      # Conflict
    "pressure",      # Pressure
    "tension",       # Narrative Tension
    "relationship",  # Relationship Movement
    "goal",          # Goal Advancement
    "character",     # Character Change
]

# 评分范围
SCORE_MIN = 1.0
SCORE_MAX = 5.0
SCORE_STEP = 0.5

# Engagement 维度（前 5 个）
ENGAGEMENT_DIMS = DIMENSIONS[:5]  # dialogue, interaction, conflict, pressure, tension

# Progression 维度（后 3 个）
PROGRESSION_DIMS = DIMENSIONS[5:]  # relationship, goal, character


# ========== 冲突关键词（外部冲突） ==========
CONFLICT_KEYWORDS = {
    "拒绝", "阻止", "威胁", "争夺", "怀疑", "质问",
    "反对", "背叛", "挑衅", "对抗", "否认", "驱逐",
    "不信", "质疑", "揭穿", "对峙", "阻拦", "驱逐",
    "斩杀", "围攻", "伏击", "偷袭", "困住", "封印"
}

# ========== 压力关键词（紧迫性与代价） ==========
PRESSURE_KEYWORDS = {
    "必须", "否则", "不得不", "只能", "如果……就……",
    "三日内", "最后", "紧迫", "来不及", "晚了",
    "代价", "牺牲", "换取", "不可逆", "永远"
}

# ========== 决策/认知变化关键词（用于 Character Change 检测） ==========
COGNITIVE_SHIFT_KEYWORDS = {
    "发现", "意识到", "原来", "终于明白", "真相",
    "怀疑", "不再相信", "开始相信", "改变看法",
    "理解", "领悟", "看透", "识破", "恍然"
}

BEHAVIOR_CHANGE_KEYWORDS = {
    "决定", "选择", "改", "换", "放弃", "转为",
    "下定决心", "发誓", "承诺", "再也不会"
}

IDENTITY_SHIFT_KEYWORDS = {
    "不再是", "成为", "叛", "立誓", "重生", "斩断",
    "告别过去", "脱胎换骨"
}

# ========== 悬念关键词（Tension） ==========
TENSION_KEYWORDS = {
    "然而", "忽然", "竟然", "却", "未解", "发现",
    "异常", "不对劲", "震惊", "原来", "难道",
    "未料", "骤变", "暗藏", "转机", "究竟"
}

# ========== 目标重定义关键词（Goal Advancement） ==========
GOAL_REDEFINE_KEYWORDS = {
    "目标变了", "不再是", "改为", "转向", "新任务",
    "真正的目的", "关键", "秘密", "真相"
}


# ========== 评分锚点（用于将原始特征映射到 1-5 分） ==========
# Dialogue Richness: 基于潜台词密度和博弈强度
DIALOGUE_ANCHORS = {
    1: "纯信息交换，无博弈",
    2: "有观点分歧但无交锋",
    3: "有明确立场冲突",
    4: "冲突中带有个性/潜台词",
    5: "对话改变角色关系或行动",
}

# Interaction: 基于角色间行为影响的强度
INTERACTION_ANCHORS = {
    1: "同处一场景但无相互影响",
    2: "单向观察或简单回应",
    3: "双向言语/行动回应",
    4: "一方迫使另一方做出选择",
    5: "互动导致角色状态/关系发生质变",
}

# Conflict: 基于阻碍的强度
CONFLICT_ANCHORS = {
    1: "无阻碍，目标顺利达成",
    2: "轻微外部阻力，轻易克服",
    3: "明确的外部对手或规则",
    4: "叠加的内部/外部冲突",
    5: "多重压迫（外部+内部+巨大代价）",
}

# Pressure: 基于代价×紧迫性
PRESSURE_ANCHORS = {
    1: "无紧迫感，无代价",
    2: "有时间限制",
    3: "时间限制 + 资源不足",
    4: "时间 + 资源 + 道德/情感代价",
    5: "多重不可逆选择",
}

# Narrative Tension: 基于未来不确定性
TENSION_ANCHORS = {
    1: "一切问题已解决，无后续钩子",
    2: "有轻微未解细节",
    3: "一个明确未回答的问题",
    4: "叠加的未解问题和即将到来的危机",
    5: "未来完全不可预测",
}

# Relationship Movement: 基于关系拓扑变化
RELATIONSHIP_ANCHORS = {
    1: "关系无变化",
    2: "数值微调，关系性质不变",
    3: "关系性质发生单次转变",
    4: "多重关系同时质变",
    5: "关系质变导致故事方向改变",
}

# Goal Advancement: 基于目标状态空间变化
GOAL_ANCHORS = {
    1: "目标状态无变化",
    2: "获得新信息，但目标路径未变",
    3: "目标路径发生变化",
    4: "目标优先级发生变化",
    5: "目标本身被替换或否定",
}

# Character Change: 基于决策模型变化
CHARACTER_ANCHORS = {
    1: "无变化",
    2: "产生怀疑",
    3: "认知模型改变（世界观重构）",
    4: "行为准则改变",
    5: "身份认同改变",
}