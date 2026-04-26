# src/model_router.py
"""
Model Router - Selects the appropriate LLM model based on task type and content.
"""

from typing import Dict, Any, Optional

# 模型配置（根据实际 llama.cpp 返回的 id 填写）
MODELS = {
    "default": "Qwen3.6-35B-A3B-UD-Q5_K_M",
    "fallback": "DeepSeek-R1-Distill-Qwen-32B-Q5_K_M",
    "code": "Qwen3.6-35B-A3B-UD-Q5_K_M",
    "research": "Qwen3.6-35B-A3B-UD-Q5_K_M",
    "validate": "Qwen3.6-35B-A3B-UD-Q5_K_M",   # 可换成小模型提速Qwen3.6-27B-Q5_K_M
    "plan": "Qwen3.6-35B-A3B-UD-Q5_K_M",
    "fast": "Qwen3.5-9B-Q4_K_M",               # 简单任务用小模型
}

class ModelRouter:
    """Simple rule-based model router."""

    def __init__(self, custom_rules: Optional[Dict[str, str]] = None):
        self.rules = custom_rules or MODELS

    def select(self, task_type: str, user_input: str = "") -> str:
        """
        Select model based on task type and optionally user input.
        Args:
            task_type: 'code', 'research', 'validate', 'plan', 'default'
            user_input: User request text, used for more granular routing.
        Returns:
            Model name string.
        """
        # 特殊处理：如果用户输入包含代码相关关键词，强制使用 code 模型
        code_keywords = ["写代码", "实现", "函数", "def ", "class ", "编写", "生成代码", "python", "斐波那契", "算法"]
        if any(kw in user_input for kw in code_keywords):
            return self.rules.get("code", self.rules["default"])
        return self.rules.get(task_type, self.rules["default"])
    

    def get_fallback(self) -> str:
        """Return fallback model in case primary fails."""
        return self.rules["fallback"]

# 全局单例（可选）
_router = None

def get_router() -> ModelRouter:
    global _router
    if _router is None:
        _router = ModelRouter()
    return _router