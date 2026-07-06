"""
Writing Quality Metrics - 纯观测器（离线实验版）
"""

import re
from typing import Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class WritingMetrics:
    total_chars: int = 0
    total_sentences: int = 0
    dialogue_chars: int = 0
    dialogue_blocks: int = 0
    inner_monologue_blocks: int = 0
    sensory_words: int = 0
    sentence_lengths: List[int] = field(default_factory=list)

    @property
    def dialogue_ratio(self) -> float:
        return self.dialogue_chars / max(self.total_chars, 1)

    @property
    def inner_monologue_density(self) -> float:
        return self.inner_monologue_blocks / max(self.total_sentences, 1)

    @property
    def sensory_density(self) -> float:
        return self.sensory_words / max(self.total_chars, 1)

    @property
    def avg_sentence_length(self) -> float:
        if not self.sentence_lengths:
            return 0.0
        return sum(self.sentence_lengths) / len(self.sentence_lengths)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_chars": self.total_chars,
            "total_sentences": self.total_sentences,
            "dialogue_ratio": round(self.dialogue_ratio, 4),
            "dialogue_blocks": self.dialogue_blocks,
            "inner_monologue_density": round(self.inner_monologue_density, 4),
            "inner_monologue_blocks": self.inner_monologue_blocks,
            "sensory_density": round(self.sensory_density, 4),
            "sensory_words": self.sensory_words,
            "avg_sentence_length": round(self.avg_sentence_length, 2),
        }


class MetricsObserver:
    SENSORY_WORDS = {
        "视觉": ["看", "见", "望", "凝视", "注视", "扫", "瞥", "盯", "瞪", "光", "影", "色", "亮", "暗", "明", "昏", "闪烁", "映", "照"],
        "听觉": ["听", "闻", "声", "音", "响", "鸣", "喧", "寂", "静", "轰", "震", "回", "荡", "沙沙", "哗啦", "叮当", "低语", "呼喊", "叹息"],
        "触觉": ["触", "摸", "抚", "按", "压", "握", "捏", "擦", "滑", "凉", "冷", "寒", "冰", "温", "暖", "热", "烫", "湿", "干", "硬", "软", "柔", "麻", "酥"],
        "嗅觉": ["嗅", "闻", "香", "臭", "腥", "腐", "熏", "芳", "味", "气息", "芬芳", "幽香", "腥气", "药香"],
        "味觉": ["尝", "味", "甜", "苦", "辣", "咸", "酸", "涩", "鲜", "滋味"],
    }

    INNER_MONOLOGUE_PATTERNS = [
        r'心想[：:，]',
        r'暗想[：:，]',
        r'转念[：:，]',
        r'暗道[：:，]',
        r'心里[想道]',
        r'心中[想道]',
        r'暗自[想道]',
        r'不由[得想]',
        r'忽然[想道]',
        r'顿时[想道]',
        r'不禁[想道]',
        r'[（(][^）)]*?[）)]',
    ]

    @classmethod
    def observe(cls, text: str) -> WritingMetrics:
        if not text or len(text.strip()) < 10:
            return WritingMetrics()
        metrics = WritingMetrics()
        metrics.total_chars = len(text)
        sentences = re.split(r'[。！？；\n]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        metrics.total_sentences = len(sentences)
        metrics.sentence_lengths = [len(s) for s in sentences]
        dialogue_matches = re.findall(r'[「『"“][^」』"“]*[」』"“]', text)
        metrics.dialogue_chars = sum(len(m) for m in dialogue_matches)
        metrics.dialogue_blocks = len(dialogue_matches)
        inner_count = 0
        for pattern in cls.INNER_MONOLOGUE_PATTERNS:
            inner_count += len(re.findall(pattern, text))
        metrics.inner_monologue_blocks = inner_count
        all_sensory_words = set()
        for category, words in cls.SENSORY_WORDS.items():
            all_sensory_words.update(words)
        sensory_total = 0
        for word in all_sensory_words:
            sensory_total += text.count(word)
        metrics.sensory_words = sensory_total
        return metrics

    @classmethod
    def compare(cls, original: str, rewritten: str) -> Dict[str, Any]:
        orig = cls.observe(original)
        new = cls.observe(rewritten)
        return {
            "original": orig.to_dict(),
            "rewritten": new.to_dict(),
            "deltas": {
                "dialogue_ratio": new.dialogue_ratio - orig.dialogue_ratio,
                "inner_monologue_density": new.inner_monologue_density - orig.inner_monologue_density,
                "sensory_density": new.sensory_density - orig.sensory_density,
                "avg_sentence_length": new.avg_sentence_length - orig.avg_sentence_length,
                "total_chars": new.total_chars - orig.total_chars,
            }
        }

    @classmethod
    def generate_report(cls, original: str, rewritten: str) -> str:
        compare = cls.compare(original, rewritten)
        lines = [
            "## 写作指标对比报告\n",
            "### 原文",
            f"- 对话占比: {compare['original']['dialogue_ratio']*100:.1f}%",
            f"- 内心活动密度: {compare['original']['inner_monologue_density']*100:.1f}%",
            f"- 感官词密度: {compare['original']['sensory_density']*100:.1f}%",
            f"- 平均句长: {compare['original']['avg_sentence_length']:.1f} 字符",
            f"- 总字符数: {compare['original']['total_chars']}",
            "",
            "### 改写后",
            f"- 对话占比: {compare['rewritten']['dialogue_ratio']*100:.1f}%",
            f"- 内心活动密度: {compare['rewritten']['inner_monologue_density']*100:.1f}%",
            f"- 感官词密度: {compare['rewritten']['sensory_density']*100:.1f}%",
            f"- 平均句长: {compare['rewritten']['avg_sentence_length']:.1f} 字符",
            f"- 总字符数: {compare['rewritten']['total_chars']}",
            "",
            "### 变化",
            f"- 对话占比: {compare['deltas']['dialogue_ratio']*100:+.1f}%",
            f"- 内心活动密度: {compare['deltas']['inner_monologue_density']*100:+.1f}%",
            f"- 感官词密度: {compare['deltas']['sensory_density']*100:+.1f}%",
            f"- 平均句长: {compare['deltas']['avg_sentence_length']:+.1f} 字符",
            f"- 总字符数: {compare['deltas']['total_chars']:+d}",
        ]
        return "\n".join(lines)