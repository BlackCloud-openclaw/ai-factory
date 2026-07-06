# src/config/character_config.py
"""
角色配置加载器 - 唯一配置入口

设计原则：
1. 使用 id 作为系统标识，name 作为显示名称
2. 通过 tags 进行角色分类，不依赖特定 id
3. 支持别名（aliases）用于实体消歧
4. 使用 lru_cache 缓存，支持热加载
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from functools import lru_cache
import yaml
from pathlib import Path


@dataclass
class CharacterIdentity:
    """角色身份"""
    id: str
    name: str
    role: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def has_tag(self, tag: str) -> bool:
        return tag in self.tags

    def is_main(self) -> bool:
        return "main" in self.tags


@dataclass
class ArtifactIdentity:
    """道具身份"""
    id: str
    name: str
    aliases: List[str] = field(default_factory=list)
    significance: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def matches(self, text: str) -> bool:
        """检查文本是否指向本实体（包含别名匹配）"""
        if not text:
            return False
        text_lower = text.lower().strip()
        if self.name.lower() == text_lower:
            return True
        for alias in self.aliases:
            if alias.lower() == text_lower:
                return True
            # 也支持部分匹配（如 "古玉" 匹配 "古玉碎片"）
            if alias.lower() in text_lower:
                return True
        return False

    def has_tag(self, tag: str) -> bool:
        return tag in self.tags


@dataclass
class CharacterConfig:
    """角色配置 - 唯一配置入口"""
    characters: Dict[str, CharacterIdentity] = field(default_factory=dict)
    artifacts: Dict[str, ArtifactIdentity] = field(default_factory=dict)

    # 反向索引
    _name_to_id: Dict[str, str] = field(default_factory=dict, repr=False)
    _alias_to_id: Dict[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._build_index()

    def _build_index(self):
        """构建反向索引：name -> id, alias -> id"""
        self._name_to_id.clear()
        self._alias_to_id.clear()

        for char_id, char in self.characters.items():
            self._name_to_id[char.name] = char_id

        for art_id, art in self.artifacts.items():
            self._name_to_id[art.name] = art_id
            for alias in art.aliases:
                self._alias_to_id[alias] = art_id

    # ========== 角色查询 ==========

    def get_character(self, char_id: str) -> Optional[CharacterIdentity]:
        return self.characters.get(char_id)

    def get_character_by_name(self, name: str) -> Optional[CharacterIdentity]:
        """通过名称查找角色"""
        char_id = self._name_to_id.get(name)
        if char_id:
            return self.characters.get(char_id)
        # 遍历匹配（兜底）
        for char in self.characters.values():
            if char.name == name:
                return char
        return None

    def get_character_id_by_name(self, name: str) -> Optional[str]:
        """通过名称获取角色 ID"""
        char_id = self._name_to_id.get(name)
        if char_id:
            return char_id
        # 遍历匹配（兜底）
        for char in self.characters.values():
            if char.name == name:
                return char.id
        return None

    def get_character_name(self, char_id: str) -> str:
        char = self.get_character(char_id)
        return char.name if char else char_id

    # ========== 主角查询（基于 tags） ==========

    def get_main_character(self) -> Optional[CharacterIdentity]:
        """通过 tags 查找主角，优先 main 标签，否则 fallback 到 id='protagonist'"""
        for char in self.characters.values():
            if char.is_main():
                return char
        # fallback：兼容旧配置
        return self.characters.get("protagonist")

    def get_main_character_id(self) -> str:
        char = self.get_main_character()
        return char.id if char else "protagonist"

    def get_main_character_name(self) -> str:
        char = self.get_main_character()
        return char.name if char else "主角"

    def get_character_id_by_tag(self, tag: str) -> List[str]:
        """获取所有拥有指定 tag 的角色 ID 列表"""
        return [char.id for char in self.characters.values() if char.has_tag(tag)]

    # ========== 道具查询 ==========

    def get_artifact(self, artifact_id: str) -> Optional[ArtifactIdentity]:
        return self.artifacts.get(artifact_id)

    def get_artifact_name(self, artifact_id: str) -> str:
        art = self.get_artifact(artifact_id)
        return art.name if art else artifact_id

    def resolve_artifact(self, text: str) -> Optional[str]:
        """通过名称或别名解析 artifact ID"""
        if not text:
            return None
        text_lower = text.lower().strip()

        # 1. 精确匹配名称
        if text_lower in self._name_to_id:
            return self._name_to_id[text_lower]

        # 2. 别名匹配
        if text_lower in self._alias_to_id:
            return self._alias_to_id[text_lower]

        # 3. 遍历匹配（兜底）
        for art in self.artifacts.values():
            if art.matches(text):
                return art.id
        return None

    def get_artifact_id_by_tag(self, tag: str) -> List[str]:
        """获取所有拥有指定 tag 的道具 ID 列表"""
        return [art.id for art in self.artifacts.values() if art.has_tag(tag)]

    # ========== 通用 ==========

    def get_all_character_ids(self) -> List[str]:
        return list(self.characters.keys())

    def get_all_character_names(self) -> List[str]:
        return [char.name for char in self.characters.values()]


# ============ 全局加载器（支持缓存清理） ============

_config: Optional[CharacterConfig] = None


def load_character_config() -> CharacterConfig:
    """加载角色配置（单例缓存）"""
    global _config
    if _config is not None:
        return _config

    config_path = Path("config/xianxia/character.yaml")
    if not config_path.exists():
        # 降级默认值
        _config = CharacterConfig(
            characters={
                "protagonist": CharacterIdentity(
                    id="protagonist",
                    name="林逸",
                    tags=["main"]
                )
            }
        )
        return _config

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    characters = {}
    for char_data in data.get("characters", []):
        char = CharacterIdentity(
            id=char_data["id"],
            name=char_data["name"],
            role=char_data.get("role"),
            tags=char_data.get("tags", []),
        )
        characters[char.id] = char

    artifacts = {}
    for art_data in data.get("artifacts", []):
        art = ArtifactIdentity(
            id=art_data["id"],
            name=art_data["name"],
            aliases=art_data.get("aliases", []),
            significance=art_data.get("significance"),
            tags=art_data.get("tags", []),
        )
        artifacts[art.id] = art

    _config = CharacterConfig(characters=characters, artifacts=artifacts)
    return _config


def reload_character_config() -> CharacterConfig:
    """清除缓存并重新加载（用于热加载）"""
    global _config
    _config = None
    return load_character_config()


# ============ 便捷函数（推荐业务层使用） ============

def get_main_character_id() -> str:
    """获取主角 ID（首选业务接口）"""
    return load_character_config().get_main_character_id()


def get_main_character_name() -> str:
    """获取主角名称（仅用于显示）"""
    return load_character_config().get_main_character_name()


def get_character_name(char_id: str) -> str:
    """获取角色名称（仅用于显示）"""
    return load_character_config().get_character_name(char_id)


def get_character_id_by_name(name: str) -> Optional[str]:
    """通过名称获取角色 ID"""
    return load_character_config().get_character_id_by_name(name)


def get_artifact_name(artifact_id: str) -> str:
    """获取道具名称（仅用于显示）"""
    return load_character_config().get_artifact_name(artifact_id)


def resolve_artifact(text: str) -> Optional[str]:
    """通过名称或别名解析道具 ID"""
    return load_character_config().resolve_artifact(text)