from dataclasses import dataclass
from typing import List, Optional, Union
import logging

from .models import MissingContractChange, ContractSeverity
from .feedback import ValidationFeedbackCompiler
from src.writing.runtime.enforcement_mode import EnforcementMode
from src.writing.runtime.validation_policy import ValidationPolicy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetryDecision:
    should_retry: bool
    writing_feedback: str = ""
    next_retry_count: int = 0
    reason: str = ""


class ContractRetryController:
    """Contract 缺失重试决策控制器。

    职责：
    - 根据 MissingContractChange 列表、重试次数和 ValidationPolicy 决定是否重试。
    - 若决定重试，通过 ValidationFeedbackCompiler 生成反馈文本。
    - 仅返回 RetryDecision，不修改任何状态，不记录 CONTRACT_RETRY_TRIGGERED。
    """

    def __init__(
        self,
        feedback_compiler: Optional[ValidationFeedbackCompiler] = None,
    ):
        self.feedback_compiler = feedback_compiler or ValidationFeedbackCompiler(max_items=3)

    def _normalize(
        self,
        missing_changes: List[Union[MissingContractChange, dict]],
    ) -> List[MissingContractChange]:
        """将混合输入（dict/对象）归一化为 MissingContractChange 列表。

        跳过无效项，确保 Controller 不会因输入格式问题崩溃。
        """
        normalized = []
        for item in missing_changes:
            if isinstance(item, MissingContractChange):
                normalized.append(item)
            elif isinstance(item, dict):
                try:
                    normalized.append(MissingContractChange.from_dict(item))
                except Exception as e:
                    logger.warning(
                        "Failed to normalize missing contract change: %s, error: %s",
                        item,
                        e,
                    )
            else:
                logger.warning(
                    "Skipping invalid missing change type: %s",
                    type(item),
                )
        return normalized

    def decide(
        self,
        missing_changes: List[Union[MissingContractChange, dict]],
        retry_count: int,
        policy: ValidationPolicy,
    ) -> RetryDecision:
        """执行重试决策。

        决策顺序：
        1. 归一化输入。
        2. 检查是否存在 BLOCKING 级别缺失。
        3. 检查 enforcement_mode 是否允许重试。
        4. 检查重试预算是否耗尽。
        5. 允许重试 → 编译反馈。
        """
        changes = self._normalize(missing_changes)
        if not changes:
            return RetryDecision(should_retry=False, reason="No valid missing changes")

        has_blocking = any(c.severity == ContractSeverity.BLOCKING for c in changes)
        if not has_blocking:
            return RetryDecision(
                should_retry=False,
                reason="Only WARNING-level missing changes, no BLOCKING",
            )

        if policy.enforcement_mode not in (EnforcementMode.RETRY, EnforcementMode.STRICT):
            return RetryDecision(
                should_retry=False,
                reason=f"Enforcement mode {policy.enforcement_mode.value} does not allow retry",
            )

        if retry_count >= policy.max_retry:
            return RetryDecision(
                should_retry=False,
                reason=f"Retry count {retry_count} reached max {policy.max_retry}",
            )

        feedback = self.feedback_compiler.compile(changes)
        next_retry = retry_count + 1

        return RetryDecision(
            should_retry=True,
            writing_feedback=feedback,
            next_retry_count=next_retry,
            reason=f"BLOCKING missing changes, retry {next_retry}/{policy.max_retry}",
        )