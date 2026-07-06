"""
WQI V1 - 写作质量指数（离线实验版）

评分维度：
- 对话占比：25分
- 感官密度：15分
- 内心活动密度：15分
- 句长节奏：10分
- 声音多样性：20分
- 重复惩罚：15分
总分：100分
"""

import re
from typing import Dict, Any, List, Optional
from collections import Counter


class WQIV1:
    """写作质量指数 V1（仅用于离线实验）"""

    @classmethod
    def score(cls, text: str, original_text: Optional[str] = None) -> Dict[str, Any]:
        """
        计算 WQI V1
        
        Args:
            text: 待评分的文本
            original_text: 原始文本（用于计算字数倍率）
        
        Returns:
            包含总分和各项得分的字典
        """
        # 1. 基础指标
        metrics = cls._observe(text)
        
        # 2. 各维度得分
        scores = {}
        
        # 2.1 对话占比（目标 0.25-0.50）
        dialogue_ratio = metrics["dialogue_ratio"]
        if 0.25 <= dialogue_ratio <= 0.50:
            scores["dialogue"] = 25.0
        elif dialogue_ratio < 0.25:
            scores["dialogue"] = (dialogue_ratio / 0.25) * 25.0
        else:
            scores["dialogue"] = max(0, 25.0 - (dialogue_ratio - 0.50) * 100)
        scores["dialogue"] = min(25.0, max(0, scores["dialogue"]))
        
        # 2.2 感官密度（目标 ≥0.10）
        sensory = metrics["sensory_density"]
        scores["sensory"] = min(15.0, (sensory / 0.10) * 15.0)
        
        # 2.3 内心活动密度（目标 ≥0.08）
        inner = metrics["inner_monologue_density"]
        scores["inner"] = min(15.0, (inner / 0.08) * 15.0)
        
        # 2.4 句长节奏（目标 15-22）
        avg_len = metrics["avg_sentence_length"]
        if 15 <= avg_len <= 22:
            scores["rhythm"] = 10.0
        else:
            ideal = 18
            diff = abs(avg_len - ideal)
            scores["rhythm"] = max(0, 10.0 - diff * 0.5)
        
        # 2.5 声音多样性
        voice_score = cls._voice_diversity_score(text)
        scores["voice"] = voice_score * 20.0
        
        # 2.6 重复惩罚
        repetition_score = cls._repetition_score(text)
        scores["repetition"] = repetition_score * 15.0
        
        # 3. 字数倍率检查（仅当有原文时）
        length_ratio = 1.0
        if original_text:
            orig_len = len(original_text)
            new_len = len(text)
            length_ratio = new_len / max(orig_len, 1)
            if 0.8 <= length_ratio <= 2.0:
                length_ok = True
            else:
                length_ok = False
        else:
            length_ok = True
        
        # 4. 汇总
        total = sum(scores.values())
        total = min(100, max(0, total))
        
        return {
            "total": round(total, 1),
            "scores": {k: round(v, 1) for k, v in scores.items()},
            "metrics": metrics,
            "length_ratio": round(length_ratio, 2),
            "length_ok": length_ok,
        }

    @classmethod
    def _observe(cls, text: str) -> Dict[str, float]:
        """提取基础指标"""
        if not text:
            return {}
        
        total_chars = len(text)
        
        # 句子分割
        sentences = re.split(r'[。！？；\n]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        total_sentences = len(sentences)
        avg_len = sum(len(s) for s in sentences) / max(total_sentences, 1)
        
        # 对话
        dialogue_matches = re.findall(r'[「『"“][^」』"“]*[」』"“]', text)
        dialogue_chars = sum(len(m) for m in dialogue_matches)
        dialogue_ratio = dialogue_chars / max(total_chars, 1)
        
        # 内心活动
        inner_patterns = [
            r'心想[：:，]',
            r'暗想[：:，]',
            r'转念[：:，]',
            r'心里[想道]',
            r'心中[想道]',
            r'暗自[想道]',
            r'不由[得想]',
            r'忽然[想道]',
            r'[（(][^）)]*?[）)]',
        ]
        inner_count = 0
        for pattern in inner_patterns:
            inner_count += len(re.findall(pattern, text))
        inner_density = inner_count / max(total_sentences, 1)
        
        # 感官词
        sensory_words = set()
        for category in ["视觉", "听觉", "触觉", "嗅觉", "味觉"]:
            # 简化词库
            words = []
            if category == "视觉":
                words = ["看", "见", "望", "凝视", "注视", "扫", "瞥", "盯", "瞪", "光", "影", "色", "亮", "暗", "明", "昏", "闪烁", "映", "照"]
            elif category == "听觉":
                words = ["听", "闻", "声", "音", "响", "鸣", "喧", "寂", "静", "轰", "震", "回", "荡", "沙沙", "哗啦", "叮当"]
            elif category == "触觉":
                words = ["触", "摸", "抚", "按", "压", "握", "捏", "擦", "滑", "凉", "冷", "寒", "冰", "温", "暖", "热", "烫", "湿", "干"]
            elif category == "嗅觉":
                words = ["香", "臭", "腥", "腐", "熏", "芳", "味", "气息"]
            else:
                words = ["尝", "味", "甜", "苦", "辣", "咸", "酸", "涩"]
            sensory_words.update(words)
        sensory_count = sum(text.count(w) for w in sensory_words)
        sensory_density = sensory_count / max(total_chars, 1)
        
        return {
            "total_chars": total_chars,
            "total_sentences": total_sentences,
            "dialogue_ratio": dialogue_ratio,
            "sensory_density": sensory_density,
            "inner_monologue_density": inner_density,
            "avg_sentence_length": avg_len,
        }

    @classmethod
    def _voice_diversity_score(cls, text: str) -> float:
        """计算声音多样性得分（0-1）"""
        # 提取对话中的说话动词
        verb_patterns = [
            r'([^，。！？\n]{1,6})[：:]?说[道]?',
            r'([^，。！？\n]{1,6})[：:]?笑[道]?',
            r'([^，。！？\n]{1,6})[：:]?冷[笑]?',
            r'([^，。！？\n]{1,6})[：:]?沉声道',
            r'([^，。！？\n]{1,6})[：:]?低声道',
            r'([^，。！？\n]{1,6})[：:]?叹[道]?',
        ]
        speakers = set()
        for pattern in verb_patterns:
            matches = re.findall(pattern, text)
            speakers.update([m.strip() for m in matches if len(m.strip()) >= 2])
        
        # 如果只有一个说话人，多样性低
        if len(speakers) <= 1:
            return 0.1
        if len(speakers) >= 4:
            return 1.0
        return len(speakers) / 4.0

    @classmethod
    def _repetition_score(cls, text: str) -> float:
        """计算重复惩罚得分（0-1），分数越高越好（重复越少）"""
        # 提取4字短语
        words = re.findall(r'[\u4e00-\u9fff]{4}', text)
        if len(words) < 10:
            return 1.0
        
        counter = Counter(words)
        top_20 = counter.most_common(20)
        # 计算重复率：出现超过1次的短语占比
        repeated = sum(count for word, count in top_20 if count > 1)
        total = sum(count for word, count in top_20)
        repetition_rate = repeated / max(total, 1)
        
        # 重复率越高，得分越低
        return max(0, 1.0 - repetition_rate * 0.5)