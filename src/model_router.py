# src/model_router.py
from typing import List, Optional

# 每个任务类型只保留最可靠的模型（按你的环境调整）
MODEL_CANDIDATES = {
    "code": [
        "Qwen2.5-Coder-32B-Instruct-Q5_K_M",   # 代码专用
    ],
    "writing": [
        "Qwen3.6-27B-Q5_K_M",                    # 写作专用
    ],
    "research": [
        "DeepSeek-R1-Distill-Llama-70B-Q5_K_M",  # 通用，适合研究
    ],
    "validate": [
        "DeepSeek-R1-Distill-Qwen-32B-Q5_K_M",  # 轻量，用于验证
    ],
    "plan": [
        "Qwen3.6-35B-A3B-UD-Q5_K_M",
    ],
    "default": [
        "Qwen3.6-35B-A3B-UD-Q5_K_M",
    ],
    "deepseek70b": [
        "DeepSeek-R1-Distill-Llama-70B-Q5_K_M",
    ],
     "qwen122b": [
        "Qwen3.5-122B-A10B-Q4_K_M",
    ],
}

class ModelRouter:
    def __init__(self):
        self.candidates = MODEL_CANDIDATES

    def detect_task(self, user_input: str) -> str:
        lower = user_input.lower()
        if any(kw in lower for kw in ["写代码", "实现", "函数", "def ", "class ", "生成代码", "python", "算法"]):
            return "code"
        if any(kw in lower for kw in ["写小说", "写故事", "续写", "润色", "创作", "小说", "章节", "诗歌"]):
            return "writing"
        if any(kw in lower for kw in ["研究", "搜索", "分析", "什么是", "解释", "原理", "查找"]):
            return "research"
        if any(kw in lower for kw in ["验证", "检查", "确认"]):
            return "validate"
        if any(kw in lower for kw in ["计划", "规划", "拆分"]):
            return "plan"
        return "default"

    def get_candidates(self, user_input: str) -> List[str]:
        task = self.detect_task(user_input)
        return self.candidates.get(task, self.candidates["default"])

    def get_fallback(self) -> str:
        return "Qwen3.6-35B-A3B-UD-Q5_K_M"

_router: Optional[ModelRouter] = None

def get_router() -> ModelRouter:
    global _router
    if _router is None:
        _router = ModelRouter()
    return _router