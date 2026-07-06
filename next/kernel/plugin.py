from abc import ABC, abstractmethod
from typing import Dict, Any, List

class DomainPlugin(ABC):
    """题材插件抽象接口，Kernel 通过此接口获取题材相关配置"""

    @property
    @abstractmethod
    def domain_name(self) -> str:
        """题材名称，如 'xianxia', 'scifi', 'detective'"""
        pass

    @abstractmethod
    def get_rank_config(self) -> Dict[str, Any]:
        """返回 rank.yaml 内容"""
        pass

    @abstractmethod
    def get_world_rules(self) -> List[Dict[str, Any]]:
        """返回 world_rules.yaml 中的规则列表"""
        pass

    @abstractmethod
    def get_themes(self) -> List[Dict[str, Any]]:
        """返回 theme.yaml 中的主题列表"""
        pass

    @abstractmethod
    def get_conflict_keywords(self) -> List[str]:
        """返回冲突关键词列表"""
        pass

    @abstractmethod
    def get_character_config(self) -> Dict[str, Any]:
        """返回 character.yaml 内容"""
        pass

    # 可选：其他题材特定方法（如资源类型、感知规则等）