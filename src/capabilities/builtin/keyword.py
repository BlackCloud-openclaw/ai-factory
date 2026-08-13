# src/capabilities/builtin/keyword.py

from packaging.version import Version

from src.capabilities import CapabilitySpec, CapabilityMetadata


class KeywordCapability:
    """内置关键词匹配器"""
    def match(self, text: str, config: dict) -> list:
        keywords = config.get("keywords", [])
        results = []
        for kw in keywords:
            pos = text.find(kw)
            if pos != -1:
                results.append({
                    "pattern_type": "keyword",
                    "start": pos,
                    "end": pos + len(kw),
                    "text": kw,
                })
        return results


SPEC = CapabilitySpec(
    id="builtin.keyword",
    version=Version("1.0.0"),
    metadata=CapabilityMetadata(
        display_name="关键词匹配",
        description="匹配指定的关键词列表",
    ),
    config_schema={
        "type": "object",
        "properties": {
            "keywords": {"type": "array", "items": {"type": "string"}},
        },
    },
)

IMPLEMENTATION = KeywordCapability()