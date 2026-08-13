import logging
from typing import List, Union

from .models import MissingContractChange, ContractSeverity

logger = logging.getLogger(__name__)


class ValidationFeedbackCompiler:
    """将缺失的契约项编译为自然语言反馈，供 Writer 修正使用。"""

    def __init__(self, max_items: int = 3, include_severity: bool = False):
        self.max_items = max_items
        self.include_severity = include_severity

    def compile(self, missing_changes: List[Union[MissingContractChange, dict]]) -> str:
        """
        生成反馈文本。

        只包含严重性为 BLOCKING 的缺失项；若无 BLOCKING 项，则包含所有 WARNING 项（若有）。
        若没有缺失项，返回空字符串。
        """
        if not missing_changes:
            return ""

        # Normalize 输入
        normalized = []
        for item in missing_changes:
            try:
                if isinstance(item, MissingContractChange):
                    normalized.append(item)
                elif isinstance(item, dict):
                    normalized.append(MissingContractChange.from_dict(item))
                else:
                    logger.warning(
                        "Skipping invalid missing change type: %s",
                        type(item)
                    )
            except Exception as e:
                logger.warning(
                    "Failed to normalize missing change: %s, error: %s",
                    item,
                    e
                )
                continue

        if not normalized:
            return ""

        # 分离 BLOCKING 和 WARNING
        blocking = [m for m in normalized if m.severity == ContractSeverity.BLOCKING]
        warnings = [m for m in normalized if m.severity == ContractSeverity.WARNING]

        # 优先使用 blocking 项，若没有则使用 warnings
        items = blocking if blocking else warnings
        if not items:
            return ""

        # 限制数量
        if len(items) > self.max_items:
            items = items[:self.max_items]
            truncated = True
        else:
            truncated = False

        # 构建反馈
        lines = [
            "上一轮生成未能满足以下契约要求：",
            ""
        ]

        for idx, change in enumerate(items, 1):
            line = f"{idx}. {change.description}"
            if self.include_severity and change.severity == ContractSeverity.BLOCKING:
                line += " (必须修正)"
            elif self.include_severity and change.severity == ContractSeverity.WARNING:
                line += " (建议修正)"
            lines.append(line)

        if truncated:
            lines.append(f"（仅展示前 {self.max_items} 项，其余类似）")

        lines.append("")
        lines.append("请在下一次生成中明确体现上述状态变化。")

        feedback = "\n".join(lines)
        logger.info(
            "CONTRACT_FEEDBACK_GENERATED: total_missing=%d blocking=%d included=%d",
            len(normalized),
            len(blocking),
            len(items)
        )
        return feedback