"""
Capability 标识符（Phase 7）
Surface 通过引用这些 ID 声明所需能力

未来 CapabilitySpec 引入后，这里将作为 ID 与 Spec 的桥接层
"""


class Matchers:
    """模式匹配能力"""
    QUOTATION = "quotation"   # 引号检测
    KEYWORD = "keyword"       # 关键词匹配
    REGEX = "regex"           # 正则匹配


class Metrics:
    """度量能力（Phase 8 扩展）"""
    pass


class Repairs:
    """修复能力"""
    INSERT_DIALOGUE = "INSERT_DIALOGUE"
    REPLACE_SENTENCE = "REPLACE_SENTENCE"
    INSERT_AFTER = "INSERT_AFTER"
    INSERT_BEFORE = "INSERT_BEFORE"


class Triggers:
    """触发条件能力"""
    NON_COMPLIANT = "non_compliant"
    # 未来扩展