"""
PlanningContract Loader - 统一恢复与异常边界

职责：
- 从 dict 或对象恢复 PlanningContract
- 失败时抛出 ContractRecoveryError（非静默）
- 保留 Pydantic ValidationError 原始异常链
"""

import logging
from typing import Union

from src.writing.planning_contract import PlanningContract

logger = logging.getLogger(__name__)


class ContractRecoveryError(Exception):
    """PlanningContract 恢复失败异常"""
    pass


class PlanningContractLoader:
    """
    统一 Loader，确保 Writer 总能获得有效的 PlanningContract 对象。
    失败时抛出异常，绝不静默降级。
    """

    @staticmethod
    def load(data: Union[PlanningContract, dict, None]) -> PlanningContract:
        if data is None:
            raise ContractRecoveryError("PlanningContract data is None")

        if isinstance(data, PlanningContract):
            logger.critical(
                "PLANNING_CONTRACT_RECOVERY: input=PlanningContract (already object)"
            )
            return data

        if isinstance(data, dict):
            logger.critical(
                "PLANNING_CONTRACT_RECOVERY: input=dict, keys=%s, scene_id=%s",
                list(data.keys()),
                data.get("scene_id", "unknown")
            )

            try:
                contract = PlanningContract.model_validate(data)
                logger.critical(
                    "PLANNING_CONTRACT_RECOVERY: output=PlanningContract (success), scene_id=%s",
                    contract.scene_id
                )
                return contract
            except Exception as e:
                # 保留原始异常链，便于 D.5 追溯 schema 问题
                logger.exception(
                    "CONTRACT_RECOVERY_FAILED: scene_id=%s error_type=%s",
                    data.get("scene_id", "unknown"),
                    type(e).__name__,
                )
                raise ContractRecoveryError(
                    f"Failed to recover PlanningContract: {e}"
                ) from e

        raise ContractRecoveryError(
            f"Unsupported data type: {type(data).__name__}"
        )