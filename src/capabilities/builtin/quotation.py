# src/capabilities/builtin/quotation.py

import re
from packaging.version import Version

from src.capabilities import CapabilitySpec, CapabilityMetadata


class QuotationCapability:
    """内置引号匹配器"""
    def match(self, text: str, config: dict) -> list:
        patterns = config.get("patterns", [r'「.*?」', r'『.*?』', r'“.*?”', r'".*?"'])
        results = []
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                results.append({
                    "pattern_type": "quotation",
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(0),
                })
        return results


SPEC = CapabilitySpec(
    id="builtin.quotation",
    version=Version("1.0.0"),
    metadata=CapabilityMetadata(
        display_name="引号匹配",
        description="匹配中文/英文引号内容",
    ),
    config_schema={
        "type": "object",
        "properties": {
            "patterns": {"type": "array", "items": {"type": "string"}},
        },
    },
)

IMPLEMENTATION = QuotationCapability()