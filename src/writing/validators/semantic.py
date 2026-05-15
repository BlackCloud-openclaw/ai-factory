# src/writing/validators/semantic.py
"""语义验证器 - 使用 embedding 检查语义相似度（宽松模式）"""
import json
from typing import Dict, Any, Tuple, Optional, List
from .base import BaseValidator


class SemanticValidator(BaseValidator):
    """语义验证器（使用 embedding，宽松阈值）"""
    
    fatal = False
    SIMILARITY_THRESHOLD = 0.65  # 降低阈值，更宽松
    
    async def validate_async(self, text: str, context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """异步版本：检查 must_events 是否在语义上被覆盖"""
        must_events = context.get("must_events", [])
        if not must_events:
            return True, None
        
        parsed = context.get("parsed_output", {})
        scene_text = parsed.get("scene_text", text)
        
        if not scene_text or len(scene_text.strip()) < 50:
            return False, f"场景正文过短（{len(scene_text)}字符），无法判断是否包含必须事件"
        
        try:
            # 导入 embedding 生成函数（延迟导入避免循环）
            from src.writing.summarizer import generate_embedding, cosine_similarity
            
            # 分割正文为段落（取前 2000 字符，避免过长）
            short_text = scene_text[:4000]
            text_embedding_str = await generate_embedding(short_text)
            text_embedding = json.loads(text_embedding_str)
            
            missing_events = []
            for event in must_events:
                event_embedding_str = await generate_embedding(event)
                event_embedding = json.loads(event_embedding_str)
                
                similarity = cosine_similarity(event_embedding, text_embedding)
                if similarity < self.SIMILARITY_THRESHOLD:
                    missing_events.append(event)
            
            if missing_events:
                return False, f"语义缺失必须事件: {', '.join(missing_events)} (相似度 < {self.SIMILARITY_THRESHOLD})"
            
            return True, None
            
        except Exception as e:
            # Embedding 服务失败时，降级为宽松关键词匹配
            return self._fallback_keyword_check(scene_text, must_events)
    
    def _fallback_keyword_check(self, text: str, must_events: List[str]) -> Tuple[bool, Optional[str]]:
        """降级方案：关键词匹配（更宽松）"""
        # 提取每个 must_event 的核心关键词
        def extract_core_keywords(event: str) -> List[str]:
            # 常见动词和虚词
            stop_words = {'拜入', '获得', '捡到', '发现', '遇到', '进入', '完成', '通过'}
            # 取长度 >= 2 的词，过滤停用词
            words = [w for w in event if len(w) >= 2]
            # 去掉常见动词
            keywords = [w for w in words if w not in stop_words]
            if not keywords and len(event) > 4:
                # 取前 4 个字符作为关键词
                return [event[:4]]
            return keywords[:2]  # 最多 2 个关键词
        
        missing = []
        for event in must_events:
            keywords = extract_core_keywords(event)
            found = any(kw in text for kw in keywords)
            if not found:
                missing.append(event)
        
        if missing:
            return False, f"降级关键词检查缺失: {', '.join(missing)}"
        return True, None
    
    def validate(self, text: str, context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """同步版本（返回提示）"""
        return True, "Use validate_async for semantic check"


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """计算两个向量的余弦相似度"""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)