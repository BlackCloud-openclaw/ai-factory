"""
Prompt 防火墙 - 防止 LLM 输出污染系统

过滤系统提示、代码块、特殊 token 等危险内容
"""
import re
from typing import Tuple, Optional


class PromptFirewall:
    """结构化输出防火墙"""
    
    # 禁止出现的模式
    FORBIDDEN_PATTERNS = [
        r"(?i)system\s*[:：]",           # 系统指令
        r"(?i)assistant\s*[:：]",        # assistant 标记
        r"(?i)user\s*[:：]",             # user 标记
        r"```\w*\s*\n.*?\n```",          # 代码块
        r"<\?xml\s+.*?\?>",              # XML 声明
        r"<\|[^>]+\|>",                  # 特殊 token (Llama)
        r"\[INST\s*\]",                  # Llama 指令
        r"<<SYS>>",                      # Llama 系统标记
        r"<\|im_start\|>",               # ChatML 标记
        r"<\|im_end\|>",
    ]
    
    # 危险关键词（完整单词匹配）
    DANGEROUS_WORDS = [
        "ignore previous instructions",
        "you are an AI",
        "as an AI model",
        "system prompt",
        "你是一个AI",
        "忽略之前的指令",
    ]
    
    @classmethod
    def sanitize(cls, text: str) -> str:
        """清洗文本，移除危险模式"""
        if not text:
            return text
        
        original = text
        
        # 移除禁止模式
        for pattern in cls.FORBIDDEN_PATTERNS:
            text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
        
        # 移除危险关键词所在行
        lines = text.split('\n')
        filtered_lines = []
        for line in lines:
            dangerous = False
            for word in cls.DANGEROUS_WORDS:
                if word.lower() in line.lower():
                    dangerous = True
                    break
            if not dangerous:
                filtered_lines.append(line)
        text = '\n'.join(filtered_lines)
        
        # 清理多余空行
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        if text != original:
            print(f"[PromptFirewall] sanitized output (length {len(original)} -> {len(text)})")
        
        return text.strip()
    
    @classmethod
    def validate(cls, text: str) -> Tuple[bool, Optional[str]]:
        """验证文本是否通过防火墙
        
        Returns:
            (passed, error_message)
        """
        # 检查是否为空
        if not text or len(text.strip()) < 10:
            return False, "Output is empty or too short"
        
        # 检查是否包含禁止模式
        for pattern in cls.FORBIDDEN_PATTERNS:
            if re.search(pattern, text, re.DOTALL | re.IGNORECASE):
                return False, f"Blocked pattern: {pattern[:50]}"
        
        # 检查危险关键词
        for word in cls.DANGEROUS_WORDS:
            if word.lower() in text.lower():
                return False, f"Blocked keyword: {word}"
        
        return True, None