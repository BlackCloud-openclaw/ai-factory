# src/writing/prompt/bundle.py

from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass(frozen=True)
class PromptSection:
    section_id: str
    renderer: str
    version: str
    priority: int
    content: str
    consumed_fields: List[str]
    
    @property
    def chars(self) -> int:
        return len(self.content)
    
    @property
    def estimated_tokens(self) -> int:
        return int(self.chars * 0.4)


@dataclass(frozen=True)
class PromptManifest:
    ir_schema: str
    renderer_versions: Dict[str, str]
    generation_profile: str
    tokenizer: Optional[str] = None
    language: str = "zh"


@dataclass(frozen=True)
class PromptBundle:
    system_prompt: str
    sections: List[PromptSection]
    manifest: PromptManifest
    schema_version: str = "1.0"
    
    @property
    def total_chars(self) -> int:
        return len(self.system_prompt) + sum(s.chars for s in self.sections)
    
    @property
    def total_tokens(self) -> int:
        return int(self.total_chars * 0.4)