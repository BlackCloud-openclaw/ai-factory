"""
B2-1B: RealmAuthority 真实实现

基于 Realm 枚举顺序判断境界跃迁合法性。
v1 规则：
- 同境界内 stage 提升 → 允许 (1-9)
- 同境界内同 stage → 允许 (NO-OP)
- 同境界内 stage 倒退 → 阻断
- 相邻大境界提升 → 允许
- 跨 2+ 大境界 → 阻断
- 倒退回低境界 → 阻断
- 未知 Realm / 无效 stage → 阻断
- 法则顿悟跨境 → v1 暂不支持
- 故事阶段不影响 → 不参与判断
"""

from typing import Optional, Dict, Any, Tuple
import logging

from src.writing.world_state import Realm

logger = logging.getLogger(__name__)

# 境界顺序权威来源：Realm Enum 本身（单一真实来源）
REALM_ORDER: list[Realm] = list(Realm)

# 境界名称到 Realm 的映射
REALM_NAME_TO_ENUM: dict[str, Realm] = {r.value: r for r in REALM_ORDER}

# 境界索引缓存
REALM_INDEX: dict[Realm, int] = {r: i for i, r in enumerate(REALM_ORDER)}


def get_realm_index(realm: Realm) -> int:
    """获取境界在顺序中的索引"""
    return REALM_INDEX.get(realm, -1)


def get_realm_by_name(name: str) -> Optional[Realm]:
    """通过名称获取 Realm 枚举"""
    return REALM_NAME_TO_ENUM.get(name)


class DefaultRealmAuthority:
    """
    默认境界权威实现 — 基于 Realm 枚举顺序。
    """

    def can_transition(
        self,
        actor: str,
        from_realm: str,
        from_stage: int,
        to_realm: str,
        to_stage: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """
        判断境界跃迁是否合法。
        """
        # 1. 解析 Realm 枚举
        try:
            from_enum = Realm(from_realm)
            to_enum = Realm(to_realm)
        except ValueError as e:
            logger.warning(f"[RealmAuthority] 未知境界名称: {e}")
            return False, f"未知境界: {from_realm} 或 {to_realm}"

        # 2. 获取索引
        from_idx = get_realm_index(from_enum)
        to_idx = get_realm_index(to_enum)

        if from_idx == -1:
            return False, f"未知境界: {from_realm}"
        if to_idx == -1:
            return False, f"未知境界: {to_realm}"

        # 3. 统一验证 stage 范围（1-9）
        if not (1 <= from_stage <= 9):
            return False, f"无效的当前小境界: {from_stage}"
        if not (1 <= to_stage <= 9):
            return False, f"无效的目标小境界: {to_stage}"

        # 4. 检查是否倒退大境界
        if to_idx < from_idx:
            return False, f"不能倒退境界 ({from_realm} → {to_realm})"

        gap = to_idx - from_idx

        # 5. 同境界内
        if gap == 0:
            if to_stage == from_stage:
                return True, f"同境界同层 ({from_realm}{from_stage}层 → {to_realm}{to_stage}层) [NO-OP]"
            elif to_stage > from_stage:
                return True, f"同境界内小境界提升 ({from_realm}{from_stage}层 → {to_realm}{to_stage}层)"
            else:
                return False, f"同境界内不能倒退小境界 ({from_realm}{from_stage}层 → {to_realm}{to_stage}层)"

        # 6. 相邻大境界提升
        if gap == 1:
            return True, f"相邻大境界突破 ({from_realm} → {to_realm})"

        # 7. 跨 2+ 大境界 → 阻断
        return False, f"不能跨越多重大境界 ({from_realm} → {to_realm}, gap={gap})"


def get_default_realm_authority() -> DefaultRealmAuthority:
    """获取默认 RealmAuthority 实例（工厂方法）"""
    return DefaultRealmAuthority()