# src/writing/contracts/event_matcher.py

import logging
from typing import Dict, Any, Optional

from src.writing.events import EventType
from src.writing.planning_contract import StateChange

logger = logging.getLogger(__name__)


def _safe_number(value) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def _keyword_coverage_match(expected: str, desc: str, threshold: float = 0.5) -> bool:
    """expected 的 2-4 字关键词在 desc 中的覆盖率 >= threshold"""
    if not expected or not desc:
        return False
    import re
    kws = re.findall(r'[\u4e00-\u9fff]{2,4}', expected)
    if not kws:
        return expected in desc
    hits = sum(1 for kw in kws if kw in desc)
    return hits / len(kws) >= threshold


def _type_alias_match_static(expected: str, actual_type: str) -> bool:
    """event_matcher 内部使用的 alias 检查（不依赖 ValidatorAgent）"""
    if not actual_type:
        return False
    # 1. 先试 alias map
    from src.agents.validator import ValidatorAgent
    aliases = ValidatorAgent.FREEFORM_TYPE_ALIASES.get(expected, set())
    if actual_type in aliases:
        return True
    # 2. Phase 15.9-fix4: 整句中文 type fallback
    #    如果 actual_type 是中文长句（> 4 字），认为它是自由描述 → 允许进入分支
    #    真正的字段匹配由 _keyword_coverage_match 判断
    if len(actual_type) > 4 and any('\u4e00' <= c <= '\u9fff' for c in actual_type):
        return True
    return False


class ContractEventMatcher:
    @staticmethod
    def match(
        change: StateChange,
        event: Dict[str, Any],
        expected_types: Optional[set] = None,
    ) -> bool:
        """
        Phase 15.9-fix v3: 感知 expected_types，绕过自由 type 问题。

        Args:
            change: Contract 中的 state_change
            event: Writer 产出的 event（type 可能是自由文本）
            expected_types: 期望的枚举 type 集合（如 {"plot_flag_set"}）
                           若提供，会优先用 expected_types 判断进入哪个分支
        """
        event_type = event.get("type")
        # Phase 15.9-fix6: 拼所有字符串字段做 desc
        # Writer 的 event 结构不定：可能是 description / trigger+consequence / detail 等
        desc_parts = [
            str(v) for k, v in event.items()
            if isinstance(v, str) and k not in ("type",)
        ]
        desc = " ".join(desc_parts) + " " + (event_type or "")

        # 决定要进入哪个分支：优先 expected_types，回退 event_type
        types_to_check = expected_types or ({event_type} if event_type else set())

        for expected in types_to_check:
            # 检查 event 是否"算作"该 expected 类型
            if not (
                event_type == expected
                or _type_alias_match_static(expected, event_type)
            ):
                continue

            # 进入对应分支
            if expected == EventType.PLOT_FLAG_SET.value:
                flag = event.get("flag") or event.get("name")
                if flag is None:
                    if _keyword_coverage_match(change.name or "", desc):
                        return True
                    continue
                if flag == change.name and event.get("value") == change.value:
                    return True

            elif expected == EventType.ITEM_ACQUIRE.value:
                actor = event.get("actor")
                item = event.get("item")
                if item is None:
                    if _keyword_coverage_match(change.item or "", desc):
                        return True
                    continue
                if actor == change.actor and item == change.item:
                    return True

            elif expected == EventType.LOCATION_ENTER.value:
                location = event.get("location")
                if location is None:
                    if _keyword_coverage_match(change.location or "", desc):
                        return True
                    continue
                if event.get("actor") == change.actor and location == change.location:
                    return True

            elif expected == EventType.REALM_UPGRADE.value:
                realm = event.get("to_major_realm")
                if realm is None:
                    if _keyword_coverage_match(change.to_major_realm or "", desc):
                        return True
                    continue
                if (event.get("actor") == change.actor
                    and realm == change.to_major_realm
                    and event.get("to_minor_stage") == change.to_minor_stage):
                    return True

            elif expected == EventType.RELATIONSHIP_CHANGE.value:
                to_char = event.get("to_char")
                if to_char is None:
                    if _keyword_coverage_match(change.to_char or "", desc):
                        return True
                    continue
                event_delta = _safe_number(event.get("delta"))
                change_delta = _safe_number(change.delta)
                if (event.get("from_char") == change.from_char
                    and to_char == change.to_char
                    and abs(event_delta - change_delta) <= 5):
                    return True

            elif expected == EventType.DISCOVERY.value:
                discovery = event.get("discovery") or event.get("name")
                name_match = discovery == change.name
                if not name_match:
                    name_match = _keyword_coverage_match(change.name or "", desc)
                if not name_match:
                    continue
                if getattr(change, "actor", None) is not None:
                    discoverer = event.get("discoverer")
                    if discoverer is not None and discoverer != change.actor:
                        continue
                return True

        return False