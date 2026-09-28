"""
B2-1A: Contract Sanity Guard

职责：
- 检查 PlanningContract 是否与当前 WorldState 冲突
- 输出结构化 SanityResult

不负责：
- 判断 realm transition 的具体业务规则（由 B2-1B RealmAuthority 提供）
- 修改 Contract / WorldState
- 调用 LLM / Planner
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Protocol, runtime_checkable, Union
from enum import Enum
from datetime import datetime
import logging

from src.writing.planning_contract import PlanningContract, StateChange
from src.writing.world_state import WorldState

logger = logging.getLogger(__name__)


class ViolationCode(str, Enum):
    """违规代码（第一版）"""
    INVALID_REALM_TRANSITION = "INVALID_REALM_TRANSITION"
    UNKNOWN_ACTOR = "UNKNOWN_ACTOR"
    CONTRACT_PARSE_ERROR = "CONTRACT_PARSE_ERROR"


@dataclass(frozen=True)
class SanityViolation:
    """单个违规项"""
    code: ViolationCode
    description: str
    target_type: str
    actor: Optional[str] = None
    current_state: Optional[Dict[str, Any]] = None
    target_state: Optional[Dict[str, Any]] = None
    details: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class SanityResult:
    """Guard 输出"""
    valid: bool
    violations: List[SanityViolation] = field(default_factory=list)
    check_id: str = ""
    scene_id: str = ""
    contract_id: str = ""
    contract_version: str = "1.0"
    checked_at: datetime = field(default_factory=datetime.now)

    @property
    def summary(self) -> str:
        if self.valid:
            return "Contract passed sanity check"
        return f"Contract invalid: {len(self.violations)} violation(s)"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "violations": [
                {
                    "code": v.code.value,
                    "description": v.description,
                    "target_type": v.target_type,
                    "actor": v.actor,
                    "current_state": v.current_state,
                    "target_state": v.target_state,
                    "details": v.details,
                }
                for v in self.violations
            ],
            "check_id": self.check_id,
            "scene_id": self.scene_id,
            "contract_id": self.contract_id,
            "contract_version": self.contract_version,
            "checked_at": self.checked_at.isoformat(),
        }


@runtime_checkable
class RealmAuthority(Protocol):
    """
    境界权威接口 — B2-1B 实现，Guard 只调用。

    注意：B2-1A 阶段不提供真实实现，仅定义接口。
    """
    def can_transition(
        self,
        actor: str,
        from_realm: str,
        from_stage: int,
        to_realm: str,
        to_stage: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> tuple[bool, str]:
        """
        返回: (是否允许, 原因描述)
        """
        ...


class ContractSanityGuard:
    """
    Contract 合法性检查器 — 只读，无副作用
    """

    def __init__(self, realm_authority: Optional[RealmAuthority] = None):
        self._realm_authority = realm_authority
        self._check_count = 0

    def check(
        self,
        contract: Union[PlanningContract, Dict[str, Any]],
        world_state: WorldState,
        scene_id: str,
        contract_id: Optional[str] = None,
    ) -> SanityResult:
        """
        检查 Contract 是否在当前 WorldState 下合法。
        """
        self._check_count += 1
        check_id = f"sanity_{self._check_count:06d}"

        # 如果是 dict，先转换为 PlanningContract
        if isinstance(contract, dict):
            try:
                contract = PlanningContract(**contract)
            except Exception as e:
                logger.error(f"[ContractSanityGuard] Contract parse error: {e}")
                return SanityResult(
                    valid=False,
                    violations=[
                        SanityViolation(
                            code=ViolationCode.CONTRACT_PARSE_ERROR,
                            description=f"无法解析 PlanningContract: {e}",
                            target_type="contract",
                        )
                    ],
                    check_id=check_id,
                    scene_id=scene_id,
                    contract_id=contract_id or "",
                )

        violations: List[SanityViolation] = []

        for change in contract.observables.state_changes:
            change_type = self._get_change_type(change)

            if change_type == "realm_change":
                violation = self._check_realm_change(change, world_state)
                if violation:
                    violations.append(violation)

            # 第一版仅实现 realm_change，其他类型默认通过

        valid = len(violations) == 0

        if not valid:
            logger.warning(
                f"[ContractSanityGuard] {scene_id}: {len(violations)} violation(s): "
                f"{[v.code.value for v in violations]}"
            )

        # contract_id: 不伪造，只有真实存在时才保留值
        final_contract_id = contract_id or ""

        return SanityResult(
            valid=valid,
            violations=violations,
            check_id=check_id,
            scene_id=scene_id,
            contract_id=final_contract_id,
            contract_version=contract.version,
        )

    def _check_realm_change(
        self,
        change: StateChange,
        world_state: WorldState,
    ) -> Optional[SanityViolation]:
        """检查境界变化是否合法（委托给 RealmAuthority）"""
        actor = getattr(change, "actor", None)
        target_realm = getattr(change, "to_major_realm", None)
        target_stage = getattr(change, "to_minor_stage", None)

        if not actor or not target_realm:
            return None

        # 1. 检查角色是否存在
        char = world_state.get_character(actor)
        if char is None:
            return SanityViolation(
                code=ViolationCode.UNKNOWN_ACTOR,
                description=f"角色 '{actor}' 不存在于当前 WorldState",
                target_type="realm_change",
                actor=actor,
                current_state={"actor": actor},
                target_state={"to_major_realm": target_realm, "to_minor_stage": target_stage},
            )

        # 2. 如果没有 RealmAuthority，跳过合法性判断（B2-1A 降级）
        if self._realm_authority is None:
            logger.debug(
                f"[ContractSanityGuard] No RealmAuthority, skipping realm check for {actor}"
            )
            return None

        # 3. 委托给 RealmAuthority
        current_realm = char.realm.value if hasattr(char.realm, "value") else str(char.realm)
        current_stage = char.realm_level

        allowed, reason = self._realm_authority.can_transition(
            actor=actor,
            from_realm=current_realm,
            from_stage=current_stage,
            to_realm=target_realm,
            to_stage=target_stage if target_stage is not None else 0,
            context={
                "world_state": world_state,
                "global_flags": world_state.global_flags,
            },
        )

        if not allowed:
            return SanityViolation(
                code=ViolationCode.INVALID_REALM_TRANSITION,
                description=reason,
                target_type="realm_change",
                actor=actor,
                current_state={
                    "realm": current_realm,
                    "stage": current_stage,
                },
                target_state={
                    "realm": target_realm,
                    "stage": target_stage,
                },
                details={"reason": reason},
            )

        return None

    def _get_change_type(self, change: StateChange) -> str:
        """安全获取 StateChange type"""
        t = getattr(change, "type", None)
        if t is None:
            return "unknown"
        if hasattr(t, "value"):
            return t.value
        return str(t)