# src/model_router.py
from typing import List, Dict, Optional

MODEL_CANDIDATES = {
    "code": [
        "Qwen3-Coder-30B-A3B-Instruct-Q5_K_M",
        "Qwen3.6-35B-A3B-UD-Q5_K_M",
        "DeepSeek-R1-Distill-Qwen-32B-Q5_K_M",
    ],
    "writing": [
        "Qwen3.6-35B-A3B-UD-Q5_K_M",
        "DeepSeek-R1-Distill-Qwen-32B-Q5_K_M",
        "Qwen3.5-122B-A10B-Q4_K_M",  # 注意 122B 可能慢，仅作备用
    ],
    "research": [
        "DeepSeek-R1-Distill-Qwen-32B-Q5_K_M",
        "Qwen3.6-35B-A3B-UD-Q5_K_M",
    ],
    "validate": [
        "Qwen3.5-9B-Q4_K_M",
        "MiniMax-M2.7-UD-Q3_K_XL",
    ],
    "plan": [
        "Qwen3.6-35B-A3B-UD-Q5_K_M",
    ],
    "default": [
        "Qwen3.6-35B-A3B-UD-Q5_K_M",
    ],
}

class ModelRouter:
    def __init__(self):
        self.candidates = MODEL_CANDIDATES

    def detect_task(self, user_input: str) -> str:
        """根据用户输入动态判断任务类型"""
        lower = user_input.lower()
        if any(kw in lower for kw in ["写代码", "实现", "函数", "def ", "class ", "生成代码", "python", "算法"]):
            return "code"
        if any(kw in lower for kw in ["写小说", "写故事", "续写", "润色", "创作", "小说", "章节"]):
            return "writing"
        if any(kw in lower for kw in ["研究", "搜索", "分析", "什么是", "解释", "原理"]):
            return "research"
        if any(kw in lower for kw in ["验证", "检查", "确认"]):
            return "validate"
        if any(kw in lower for kw in ["计划", "规划", "拆分"]):
            return "plan"
        return "default"

    def get_candidates(self, user_input: str) -> List[str]:
        """返回该任务类型的候选模型列表（按优先级排序）"""
        task = self.detect_task(user_input)
        return self.candidates.get(task, self.candidates["default"])

    def get_fallback(self) -> str:
        """返回最终备用模型（当所有候选都失败时）"""
        return "DeepSeek-R1-Distill-Qwen-32B-Q5_K_M"

_router = None
def get_router() -> ModelRouter:
    global _router
    if _router is None:
        _router = ModelRouter()
    return _router