# src/model_router.py
from typing import List, Dict, Optional

# 任务类型到候选模型列表的映射（按优先级排序）
MODEL_CANDIDATES = {
    "code": [
        "Qwen3-Coder-30B-A3B-Instruct-Q5_K_M",
    ],
    "writing": [
        "Qwen3.6-27B-Q5_K_M",
        "Qwen3.6-35B-A3B-UD-Q5_K_M",
    ],
    "research": [
        "DeepSeek-R1-Distill-Qwen-32B-Q5_K_M",
        "Qwen3.6-35B-A3B-UD-Q5_K_M",
    ],
    "validate": [
        "Qwen3.5-9B-Q4_K_M",
    ],
    "plan": [
        "Qwen3.6-35B-A3B-UD-Q5_K_M",
    ],
    "default": [
        "Qwen3.6-35B-A3B-UD-Q5_K_M",
    ],
}

class ModelRouter:
    """简单规则模型路由器，根据用户输入的关键词判断任务类型并返回候选模型列表。"""

    def __init__(self):
        self.candidates = MODEL_CANDIDATES

    def detect_task(self, user_input: str) -> str:
        """根据用户输入动态判断任务类型。"""
        lower = user_input.lower()
        # 代码生成
        if any(kw in lower for kw in [
            "写代码", "实现", "函数", "def ", "class ", "生成代码", "python", "算法"
        ]):
            return "code"
        # 写作/创作
        if any(kw in lower for kw in [
            "写小说", "写故事", "续写", "润色", "创作", "小说", "章节", "诗歌"
        ]):
            return "writing"
        # 研究/搜索
        if any(kw in lower for kw in [
            "研究", "搜索", "分析", "什么是", "解释", "原理", "查找"
        ]):
            return "research"
        # 验证（通常由系统内部触发，用户输入很少直接包含，但保留）
        if any(kw in lower for kw in ["验证", "检查", "确认"]):
            return "validate"
        # 规划（系统内部）
        if any(kw in lower for kw in ["计划", "规划", "拆分"]):
            return "plan"
        return "default"

    def get_candidates(self, user_input: str) -> List[str]:
        """返回该任务类型的候选模型列表（按优先级排序）。"""
        task = self.detect_task(user_input)
        return self.candidates.get(task, self.candidates["default"])

    def get_fallback(self) -> str:
        """返回最终备用模型（当所有候选都失败时）。"""
        return "DeepSeek-R1-Distill-Qwen-32B-Q5_K_M"

# 全局单例
_router: Optional[ModelRouter] = None

def get_router() -> ModelRouter:
    global _router
    if _router is None:
        _router = ModelRouter()
    return _router