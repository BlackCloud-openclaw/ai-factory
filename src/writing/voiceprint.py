# src/writing/voiceprint.py
"""
角色声纹系统 - 防止角色语言风格同质化

每个角色拥有独立的语言身份，Writer 在生成对话时必须遵循。
支持通过角色 ID 或名称查找声纹，配置从 YAML 加载。

设计原则：
- 声纹绑定到角色 ID（稳定标识），而非名称（可变更）
- 支持通过名称或 ID 双向查找
- 主角声纹默认从配置加载，若 YAML 未定义则自动生成默认值
"""
from typing import Dict, List, Optional, Union
from pathlib import Path
import yaml
from pydantic import BaseModel, Field

from src.domain.identity import (
    get_main_character_id,
    get_main_character_name,
    get_character_name,
    get_character_id_by_name,
    get_character_config,
)


class CharacterVoiceprint(BaseModel):
    """角色语言身份"""
    character: str  # 可以是角色 ID 或名称（兼容期）
    speech_style: str = ""           # 总体风格描述，如"沉稳、少言、多用短句"
    common_phrases: List[str] = []   # 常用口头禅/词语
    sentence_length: str = "medium"  # short / medium / long
    vocabulary_tags: List[str] = []  # 词汇标签：文雅、粗犷、直白、含蓄
    emotional_baseline: Dict[str, float] = Field(default_factory=dict)  # 喜怒哀乐基准值
    forbidden_words: List[str] = []  # 禁止使用的词语（避免出戏）

    def get_display_name(self) -> str:
        """尝试将 character 字段解析为显示名称"""
        # 如果 character 是 ID，尝试解析为名称
        name = get_character_name(self.character)
        if name != self.character:
            return name
        # 否则原样返回（可能是直接配置的名称）
        return self.character


class VoiceprintRegistry:
    """声纹注册表 - 支持 ID 和名称双重查找"""

    def __init__(self, config_path: Optional[str] = None):
        self._voiceprints: Dict[str, CharacterVoiceprint] = {}  # key 可以是 ID 或 name
        self._id_to_name: Dict[str, str] = {}  # ID -> name 映射缓存

        # 默认声纹（fallback）
        self._default = CharacterVoiceprint(
            character="default",
            speech_style="普通叙事风格，中性、自然"
        )

        # 加载配置文件
        if config_path and Path(config_path).exists():
            self._load_from_yaml(config_path)
        else:
            # 如果配置文件不存在，使用默认配置并注册主角
            self._register_default_characters()

        # 确保主角已注册
        self._ensure_protagonist_registered()

    def _register_default_characters(self):
        """注册默认角色声纹（从配置读取主角和重要角色）"""
        config = get_character_config()

        # 为所有配置的角色创建默认声纹
        for char_id, char_identity in config.characters.items():
            if char_id not in self._voiceprints:
                # 创建默认声纹
                vp = CharacterVoiceprint(
                    character=char_id,  # 使用 ID 作为 key
                    speech_style=f"{char_identity.role or '角色'}的标准语言风格",
                    common_phrases=[],
                    sentence_length="medium",
                    vocabulary_tags=[],
                    emotional_baseline={},
                    forbidden_words=[],
                )
                self._voiceprints[char_id] = vp
                self._voiceprints[char_identity.name] = vp  # 同时用名称注册

        # 如果没有任何角色，至少注册主角
        if not self._voiceprints:
            protagonist_id = get_main_character_id()
            protagonist_name = get_main_character_name()
            vp = CharacterVoiceprint(
                character=protagonist_id,
                speech_style="少年沉稳，偶尔冲动，短句为主，说话直接",
                common_phrases=["哼", "有意思", "我偏不信", "好"],
                sentence_length="short",
                vocabulary_tags=["直白", "坚定"],
                emotional_baseline={"喜": 0.6, "怒": 0.3, "哀": 0.1, "乐": 0.5},
                forbidden_words=["呵呵", "哈哈"],
            )
            self._voiceprints[protagonist_id] = vp
            self._voiceprints[protagonist_name] = vp

    def _ensure_protagonist_registered(self):
        """确保主角声纹已注册"""
        protagonist_id = get_main_character_id()
        protagonist_name = get_main_character_name()

        # 检查是否已注册（通过 ID 或名称）
        registered = (
            protagonist_id in self._voiceprints or
            protagonist_name in self._voiceprints
        )

        if not registered:
            # 创建主角默认声纹
            vp = CharacterVoiceprint(
                character=protagonist_id,
                speech_style="少年沉稳，偶尔冲动，短句为主，说话直接",
                common_phrases=["哼", "有意思", "我偏不信", "好"],
                sentence_length="short",
                vocabulary_tags=["直白", "坚定"],
                emotional_baseline={"喜": 0.6, "怒": 0.3, "哀": 0.1, "乐": 0.5},
                forbidden_words=["呵呵", "哈哈"],
            )
            self._voiceprints[protagonist_id] = vp
            self._voiceprints[protagonist_name] = vp

    def _load_from_yaml(self, path: str):
        """从 YAML 文件加载角色声纹配置"""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        for vp_data in data.get('characters', []):
            # 检查是否有 id 字段，如果没有则使用 character 字段作为标识
            char_key = vp_data.get('id') or vp_data.get('character')
            if not char_key:
                continue

            vp = CharacterVoiceprint(
                character=char_key,
                speech_style=vp_data.get('speech_style', ''),
                common_phrases=vp_data.get('common_phrases', []),
                sentence_length=vp_data.get('sentence_length', 'medium'),
                vocabulary_tags=vp_data.get('vocabulary_tags', []),
                emotional_baseline=vp_data.get('emotional_baseline', {}),
                forbidden_words=vp_data.get('forbidden_words', []),
            )

            # 用 ID 和名称分别注册（如果能解析出名称）
            self._voiceprints[char_key] = vp

            # 尝试解析为 ID 获取名称
            name = get_character_name(char_key)
            if name != char_key:
                self._voiceprints[name] = vp

            # 如果 YAML 中直接配置了角色名作为 character 字段，也尝试反向查找 ID
            char_id = get_character_id_by_name(char_key)
            if char_id and char_id != char_key:
                self._voiceprints[char_id] = vp

        # 如果配置中没有定义主角，由 _ensure_protagonist_registered 补充

    def register(self, voiceprint: CharacterVoiceprint):
        """注册角色声纹"""
        char_key = voiceprint.character
        self._voiceprints[char_key] = voiceprint

        # 尝试解析为 ID 获取名称，双重注册
        name = get_character_name(char_key)
        if name != char_key:
            self._voiceprints[name] = voiceprint

        char_id = get_character_id_by_name(char_key)
        if char_id and char_id != char_key:
            self._voiceprints[char_id] = voiceprint

    def get(self, character: str) -> CharacterVoiceprint:
        """
        获取角色声纹

        支持通过角色 ID 或名称查找：
        - get("protagonist") -> 主角声纹
        - get("林逸") -> 主角声纹（通过名称查找）
        - get("玄老") -> 玄老声纹
        """
        if not character:
            return self._default

        # 1. 直接查找
        if character in self._voiceprints:
            return self._voiceprints[character]

        # 2. 尝试将 character 作为 ID，查找对应的名称
        name = get_character_name(character)
        if name != character and name in self._voiceprints:
            return self._voiceprints[name]

        # 3. 尝试将 character 作为名称，查找对应的 ID
        char_id = get_character_id_by_name(character)
        if char_id and char_id in self._voiceprints:
            return self._voiceprints[char_id]

        # 4. 检查是否为默认角色
        if character in ("default", "unknown", ""):
            return self._default

        # 5. 回退：创建临时声纹并注册（便于后续使用）
        logger = __import__('logging').getLogger(__name__)
        logger.warning(f"Character '{character}' not found in voiceprint registry, creating default")

        # 尝试从配置中获取角色信息
        config = get_character_config()
        char_identity = config.get_character(character) or config.get_character_by_name(character)

        if char_identity:
            vp = CharacterVoiceprint(
                character=char_identity.id,
                speech_style=f"{char_identity.role or '角色'}的标准语言风格",
            )
            self._voiceprints[char_identity.id] = vp
            self._voiceprints[char_identity.name] = vp
            return vp

        # 最终 fallback
        return self._default

    def get_by_id(self, char_id: str) -> CharacterVoiceprint:
        """通过 ID 获取声纹（推荐使用）"""
        return self.get(char_id)

    def get_by_name(self, name: str) -> CharacterVoiceprint:
        """通过名称获取声纹"""
        return self.get(name)

    def build_prompt_constraint(self, character: str) -> str:
        """
        构建用于 prompt 的语言约束文本

        Args:
            character: 角色 ID 或名称

        Returns:
            约束文本字符串，如果角色是默认角色则返回空字符串
        """
        vp = self.get(character)
        if vp.character == "default":
            return ""

        # 获取显示名称
        display_name = vp.get_display_name()
        lines = [f"【{display_name}的语言风格约束】"]

        if vp.speech_style:
            lines.append(f"整体风格：{vp.speech_style}")
        if vp.common_phrases:
            lines.append(f"常用口头禅/词语：{', '.join(vp.common_phrases)}")
        if vp.sentence_length != "medium":
            length_desc = "短句为主" if vp.sentence_length == "short" else "长句为主"
            lines.append(f"句子长度偏好：{length_desc}")
        if vp.forbidden_words:
            lines.append(f"禁止使用：{', '.join(vp.forbidden_words)}")
        lines.append("注意：对话必须严格符合上述风格，不要与其他角色雷同。")

        return "\n".join(lines)

    def list_characters(self) -> List[str]:
        """返回所有已注册角色名（去重）"""
        seen = set()
        result = []
        for key in self._voiceprints.keys():
            if key not in seen and not key.startswith('_'):
                seen.add(key)
                # 优先显示名称
                name = get_character_name(key)
                if name != key:
                    result.append(name)
                else:
                    result.append(key)
        return list(set(result))  # 去重

    def list_ids(self) -> List[str]:
        """返回所有已注册的角色 ID"""
        ids = set()
        for key in self._voiceprints.keys():
            char_id = get_character_id_by_name(key)
            if char_id:
                ids.add(char_id)
            elif key in get_character_config().characters:
                ids.add(key)
        return list(ids)

    def get_voiceprint_for_actor(self, actor_name: str) -> CharacterVoiceprint:
        """
        为 Writer 提供角色声纹（别名方法，保持兼容）
        """
        return self.get(actor_name)

    def to_dict(self) -> Dict[str, Dict]:
        """导出所有声纹为字典（用于序列化）"""
        result = {}
        # 只导出 ID 对应的声纹（避免重复）
        for char_id in self.list_ids():
            vp = self.get_by_id(char_id)
            if vp:
                result[char_id] = vp.model_dump()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Dict]) -> "VoiceprintRegistry":
        """从字典恢复声纹注册表"""
        registry = cls()
        registry._voiceprints.clear()
        for char_key, vp_data in data.items():
            vp = CharacterVoiceprint(**vp_data)
            registry._voiceprints[char_key] = vp
            # 尝试用名称也注册
            name = get_character_name(char_key)
            if name != char_key:
                registry._voiceprints[name] = vp
        return registry