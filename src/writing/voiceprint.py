"""
角色声纹系统 - 防止角色语言风格同质化

每个角色拥有独立的语言身份，Writer 在生成对话时必须遵循。
"""
from typing import Dict, List, Optional
from pathlib import Path
import yaml
from pydantic import BaseModel, Field


class CharacterVoiceprint(BaseModel):
    """角色语言身份"""
    character: str
    speech_style: str = ""           # 总体风格描述，如"沉稳、少言、多用短句"
    common_phrases: List[str] = []   # 常用口头禅/词语
    sentence_length: str = "medium"  # short / medium / long
    vocabulary_tags: List[str] = []  # 词汇标签：文雅、粗犷、直白、含蓄
    emotional_baseline: Dict[str, float] = Field(default_factory=dict)  # 喜怒哀乐基准值
    forbidden_words: List[str] = []  # 禁止使用的词语（避免出戏）


class VoiceprintRegistry:
    """声纹注册表"""
    
    def __init__(self, config_path: Optional[str] = None):
        self._voiceprints: Dict[str, CharacterVoiceprint] = {}
        self._default = CharacterVoiceprint(
            character="default",
            speech_style="普通叙事风格，中性、自然"
        )
        if config_path and Path(config_path).exists():
            self._load_from_yaml(config_path)
    
    def _load_from_yaml(self, path: str):
        """从 YAML 文件加载角色配置"""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            for vp_data in data.get('characters', []):
                vp = CharacterVoiceprint(**vp_data)
                self._voiceprints[vp.character] = vp
    
    def register(self, voiceprint: CharacterVoiceprint):
        """注册角色声纹"""
        self._voiceprints[voiceprint.character] = voiceprint
    
    def get(self, character: str) -> CharacterVoiceprint:
        """获取角色声纹，不存在则返回默认"""
        return self._voiceprints.get(character, self._default)
    
    def build_prompt_constraint(self, character: str) -> str:
        """构建用于 prompt 的语言约束文本"""
        vp = self.get(character)
        if vp.character == "default":
            return ""
        
        lines = [f"【{character}的语言风格约束】"]
        if vp.speech_style:
            lines.append(f"整体风格：{vp.speech_style}")
        if vp.common_phrases:
            lines.append(f"常用口头禅/词语：{', '.join(vp.common_phrases)}")
        if vp.sentence_length != "medium":
            lines.append(f"句子长度偏好：{'短句为主' if vp.sentence_length == 'short' else '长句为主'}")
        if vp.forbidden_words:
            lines.append(f"禁止使用：{', '.join(vp.forbidden_words)}")
        lines.append("注意：对话必须严格符合上述风格，不要与其他角色雷同。")
        return "\n".join(lines)
    
    def list_characters(self) -> List[str]:
        """返回所有已注册角色名"""
        return list(self._voiceprints.keys())