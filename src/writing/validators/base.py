"""Validator 基类"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional


class BaseValidator(ABC):
    """验证器基类"""
    
    # 是否为致命错误（如果为 True，后续验证器不再执行）
    fatal: bool = False
    
    @abstractmethod
    def validate(self, text: str, context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        验证文本
        
        Args:
            text: 待验证的文本
            context: 上下文（包含 world_state, scene_plan, must_events 等）
            
        Returns:
            (passed, error_message)
        """
        pass