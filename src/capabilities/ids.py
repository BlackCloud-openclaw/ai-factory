# src/capabilities/ids.py
# Phase 7 遗留常量 — 向后兼容
# Phase 8+ 应使用 CapabilityRef

class Matchers:
    QUOTATION = "quotation"
    KEYWORD = "keyword"
    REGEX = "regex"


class Metrics:
    pass


class Repairs:
    INSERT_DIALOGUE = "INSERT_DIALOGUE"
    REPLACE_SENTENCE = "REPLACE_SENTENCE"
    INSERT_AFTER = "INSERT_AFTER"
    INSERT_BEFORE = "INSERT_BEFORE"


class Triggers:
    NON_COMPLIANT = "non_compliant"
