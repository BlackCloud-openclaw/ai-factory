"""
Phase 15.3 — ShadowRewriteRunner

职责：
1. 接收 Original 文本 + 生产阶段的 ValidationResult
2. 调用 Rewriter 生成 Rewrite 文本
3. 使用同一个 Validator 验证 Rewrite 文本
4. 返回 ShadowRewriteResult（不可变）

不负责：
- 采样决策（由调用方决定）
- 持久化（由 Recorder 负责）
- 重试（实验阶段直接返回结果）

E-1 修改：
- 增加 _normalize_validation_result() 函数
- submit() 入口处调用，统一提取 passed/violations
- 支持 dict 和 object 两种输入形态

A-3 修改：
- 增加 _serialize_contract() 函数
- submit() 入口处调用，序列化 contract 为 JSON-compatible dict
- 支持 Pydantic model_dump(mode="json") 和 dict
- 其他类型返回 None 并记录警告
"""

from typing import Protocol, Any, Optional, Dict, List, runtime_checkable
import logging

from .result import ShadowRewriteResult, ShadowRewriteStatus

logger = logging.getLogger(__name__)


@runtime_checkable
class Rewriter(Protocol):
    """Rewrite 能力抽象：输入原文 + 契约，输出改写后的文本。"""
    async def rewrite(self, text: str, contract: Any) -> str:
        ...


@runtime_checkable
class Validator(Protocol):
    """Validator 能力抽象：输入文本 + 契约，输出 ValidationResult。"""
    async def validate(self, text: str, contract: Any) -> Any:
        ...


# ============================================================
# E-1: 归一化生产验证结果
# ============================================================
def _normalize_validation_result(result):
    """
    统一提取 passed 和 violations，支持 dict 和 object 两种形态。

    Args:
        result: Production Validator 的输出，可能是 dict 或 Pydantic 对象

    Returns:
        (passed: bool, violations: list)
    """
    if isinstance(result, dict):
        passed = result.get("passed", False)
        # 优先使用 violations，若不存在则尝试 errors 或 missing
        violations = (
            result.get("violations")
            or result.get("errors")
            or result.get("missing")
            or []
        )
        return passed, violations
    else:
        # object 形态（Pydantic 模型或 dataclass）
        passed = getattr(result, "passed", False)
        violations = (
            getattr(result, "violations", None)
            or getattr(result, "errors", None)
            or getattr(result, "missing", None)
            or []
        )
        return passed, violations
# ============================================================


# ============================================================
# A-3: contract 序列化
# ============================================================
def _serialize_contract(contract: Any) -> Optional[Dict[str, Any]]:
    """
    将 contract 序列化为 JSON-compatible dict。

    规则：
        - Pydantic BaseModel: 使用 model_dump_json() 确保 datetime 等类型被正确序列化
        - dict: 原样返回
        - 其他: 返回 None 并记录警告（不阻塞流程）
    """
    if contract is None:
        return None

    # Pydantic v2: 使用 model_dump_json 确保 datetime 等类型被正确序列化
    if hasattr(contract, "model_dump_json") and callable(contract.model_dump_json):
        try:
            # model_dump_json 会将 datetime 转换为 ISO 格式字符串
            return json.loads(contract.model_dump_json())
        except Exception as e:
            logger.warning(f"[Shadow] Pydantic JSON serialization failed: {e}")
            return None

    # dict
    if isinstance(contract, dict):
        return contract

    # 其他类型 —— 显式失败
    logger.warning(
        f"[Shadow] Cannot serialize contract of type {type(contract).__name__}, "
        "contract_data will be NULL"
    )
    return None


class ShadowRewriteRunner:
    """
    Shadow Rewrite 执行器。

    依赖注入：
    - rewriter: 实现 Rewriter 协议（如 ConstrainedRewriteService）
    - validator: 实现 Validator 协议（生产 Validator）

    注意：Runner 不存储任何状态，每次 submit 独立执行。
    """

    def __init__(self, rewriter: Rewriter, validator: Validator):
        self._rewriter = rewriter
        self._validator = validator

    async def submit(
        self,
        scene_id: str,
        original_text: str,
        contract: Any,
        original_validation_result: Any,
        experiment_id: Optional[str] = None,
        prompt_version: Optional[str] = None,
        writer_events: Optional[List[Dict[str, Any]]] = None,  # ========== Phase 15.5 新增 ==========
    ) -> ShadowRewriteResult:
        """
        执行 Shadow Rewrite。

        Args:
            scene_id: 场景标识
            original_text: 原始文本
            contract: Planning Contract 或等效契约
            original_validation_result: 生产阶段的 ValidationResult（dict 或 object）
            experiment_id: 实验标识（用于分组）
            prompt_version: Prompt 版本标识（用于实验分组）
            writer_events: Writer 生成的结构化 events（Phase 15.5 新增）

        Returns:
            ShadowRewriteResult: 不可变结果记录
        """
        # ============================================================
        # E-1: 归一化生产验证结果
        # ============================================================
        original_passed, original_violations = _normalize_validation_result(
            original_validation_result
        )
        # ============================================================

        # ============================================================
        # A-3: 序列化 contract
        # ============================================================
        contract_data = _serialize_contract(contract)
        # ============================================================

        # 2. 执行 Rewrite
        try:
            rewritten_text = await self._rewriter.rewrite(original_text, contract)
        except TimeoutError as e:
            logger.warning(f"[Shadow] Rewrite timeout for {scene_id}: {e}")
            return ShadowRewriteResult(
                scene_id=scene_id,
                original_text=original_text,
                original_validation_passed=original_passed,
                original_violations=original_violations,
                status=ShadowRewriteStatus.TIMEOUT,
                error_message=str(e),
                experiment_id=experiment_id,
                prompt_version=prompt_version or "phase15.3.v1",
                contract_data=contract_data,
                writer_events=writer_events,  # ========== Phase 15.5 新增 ==========
            )
        except Exception as e:
            logger.warning(f"[Shadow] Rewrite LLM error for {scene_id}: {e}")
            return ShadowRewriteResult(
                scene_id=scene_id,
                original_text=original_text,
                original_validation_passed=original_passed,
                original_violations=original_violations,
                status=ShadowRewriteStatus.LLM_ERROR,
                error_message=str(e),
                experiment_id=experiment_id,
                prompt_version=prompt_version or "phase15.3.v1",
                contract_data=contract_data,
                writer_events=writer_events,  # ========== Phase 15.5 新增 ==========
            )

        # 3. 验证 Rewrite（使用同一个 Validator）
        try:
            rewritten_result = await self._validator.validate(rewritten_text, contract)
            rewritten_passed = getattr(rewritten_result, 'passed', False)
            rewritten_violations = getattr(rewritten_result, 'violations', None)

            status = (
                ShadowRewriteStatus.SUCCESS
                if rewritten_passed
                else ShadowRewriteStatus.VALIDATION_FAILED
            )

            logger.debug(
                f"[Shadow] {scene_id}: original_pass={original_passed}, "
                f"rewrite_pass={rewritten_passed}, status={status.value}"
            )

            return ShadowRewriteResult(
                scene_id=scene_id,
                original_text=original_text,
                original_validation_passed=original_passed,
                original_violations=original_violations,
                rewritten_text=rewritten_text,
                rewritten_validation_passed=rewritten_passed,
                rewritten_violations=rewritten_violations,
                status=status,
                experiment_id=experiment_id,
                prompt_version=prompt_version or "phase15.3.v1",
                contract_data=contract_data,
                writer_events=writer_events,  # ========== Phase 15.5 新增 ==========
            )

        except Exception as e:
            logger.warning(f"[Shadow] Validator error for {scene_id}: {e}")
            return ShadowRewriteResult(
                scene_id=scene_id,
                original_text=original_text,
                original_validation_passed=original_passed,
                original_violations=original_violations,
                rewritten_text=rewritten_text,
                rewritten_validation_passed=None,
                rewritten_violations=None,
                status=ShadowRewriteStatus.VALIDATOR_ERROR,
                error_message=f"Validator exception: {e}",
                experiment_id=experiment_id,
                prompt_version=prompt_version or "phase15.3.v1",
                contract_data=contract_data,
                writer_events=writer_events,  # ========== Phase 15.5 新增 ==========
            )