# src/writing/voice_memory.py
import re
import json
from collections import Counter
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field


@dataclass
class VoiceFingerprint:
    """小说整体的风格指纹"""
    avg_sentence_length: float = 0.0      # 平均每句字符数
    avg_paragraph_length: float = 0.0     # 平均每段字符数
    top_keywords: List[str] = field(default_factory=list)        # 高频关键词（前30）
    sentence_starters: List[str] = field(default_factory=list)   # 常见句子开头模式
    dialogue_ratio: float = 0.0           # 对话占比（估算，暂时不实现）
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "VoiceFingerprint":
        return cls(**data)


class VoiceMemory:
    """风格记忆管理"""
    
    # 停用词列表（简单版）
    STOP_WORDS = {
        "的", "了", "是", "在", "和", "与", "或", "但", "而", "被", "把", "让", "使", "会", "能", "可以", "将", "就", "也", "还", "都", "不", "没", "有", "这", "那", "一", "我", "你", "他", "她", "它", "们", "地", "得", "着", "过", "对", "为", "以", "到", "去", "说", "看", "听", "想", "知道", "觉得", "看见", "听见"
    }
    
    def __init__(self, novel_id: str, fingerprint_dict: Optional[Dict] = None):
        self.novel_id = novel_id
        self._fingerprint: Optional[VoiceFingerprint] = None
        if fingerprint_dict:
            self._fingerprint = VoiceFingerprint.from_dict(fingerprint_dict)
    
    @property
    def fingerprint(self) -> Optional[VoiceFingerprint]:
        return self._fingerprint
    
    def update_from_chapter(self, chapter_text: str):
        """从新章节中提取风格特征，并更新指纹（移动平均）"""
        if not chapter_text or len(chapter_text.strip()) < 100:
            return
        
        # 提取特征
        # 1. 句子长度（按中文句号、感叹号、问号、分号分割）
        sentences = re.split(r'[。！？；]', chapter_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        if not sentences:
            return
        avg_sent_len = sum(len(s) for s in sentences) / len(sentences)
        
        # 2. 段落长度（按双换行分割）
        paragraphs = chapter_text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 10]
        avg_para_len = sum(len(p) for p in paragraphs) / len(paragraphs) if paragraphs else 0
        
        # 3. 关键词（取2-4字词，排除停用词）
        words = re.findall(r'[\u4e00-\u9fff]{2,4}', chapter_text)
        keywords = [w for w in words if w not in self.STOP_WORDS]
        top_keywords = [w for w, _ in Counter(keywords).most_common(30)]
        
        # 4. 句子开头模式（前4个字符）
        starters = []
        for s in sentences[:50]:
            if len(s) > 4:
                starters.append(s[:4])
        top_starters = [s for s, _ in Counter(starters).most_common(10)]
        
        if self._fingerprint is None:
            self._fingerprint = VoiceFingerprint(
                avg_sentence_length=avg_sent_len,
                avg_paragraph_length=avg_para_len,
                top_keywords=top_keywords,
                sentence_starters=top_starters,
            )
        else:
            # 指数移动平均，让指纹逐渐适应，但不过度偏离初始风格
            alpha = 0.3  # 新样本权重
            self._fingerprint.avg_sentence_length = (1 - alpha) * self._fingerprint.avg_sentence_length + alpha * avg_sent_len
            self._fingerprint.avg_paragraph_length = (1 - alpha) * self._fingerprint.avg_paragraph_length + alpha * avg_para_len
            # 关键词合并（保留历史+新词，取并集前30）
            combined = self._fingerprint.top_keywords + top_keywords
            self._fingerprint.top_keywords = [w for w, _ in Counter(combined).most_common(30)]
            # 句子开头合并
            combined_starters = self._fingerprint.sentence_starters + top_starters
            self._fingerprint.sentence_starters = [s for s, _ in Counter(combined_starters).most_common(10)]
    
    def get_style_constraints_prompt(self) -> str:
        """生成用于 Writer prompt 的风格约束文本"""
        if not self._fingerprint:
            return ""
        fp = self._fingerprint
        lines = ["【📖 已确立的叙述风格约束（请尽量保持）】"]
        if fp.avg_sentence_length > 0:
            lines.append(f"- 平均句子长度约 {int(fp.avg_sentence_length)} 字符，避免过短或过长")
        if fp.avg_paragraph_length > 0:
            lines.append(f"- 段落长度约 {int(fp.avg_paragraph_length)} 字符，保持自然分段")
        if fp.top_keywords:
            lines.append(f"- 偏好使用的词汇：{', '.join(fp.top_keywords[:15])}")
        if fp.sentence_starters:
            lines.append(f"- 常见句子开头：{', '.join(fp.sentence_starters[:5])}")
        lines.append("注意：不要突然改变叙述风格，尽量与已生成章节保持一致。")
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        return self._fingerprint.to_dict() if self._fingerprint else {}
    
    @classmethod
    def from_compressed_state(cls, novel_id: str, compressed_state: Optional[Dict]) -> "VoiceMemory":
        if compressed_state and "voice_fingerprint" in compressed_state:
            return cls(novel_id, compressed_state["voice_fingerprint"])
        return cls(novel_id)