# src/writing/causality/validator.py

import logging
from typing import Dict, Any, List, Tuple, Optional
from .rule_engine import RuleEngine
from .predicate import Predicate
from src.writing.constraint import Constraint, ConstraintRegistry
from src.config_loader import get_xianxia_config

logger = logging.getLogger(__name__)


class CausalityValidator:
    def __init__(self, rule_engine: RuleEngine = None):
        self.rule_engine = rule_engine or RuleEngine()
        self._load_realm_order()

    def _load_realm_order(self):
        """从配置加载境界顺序，如果失败则使用默认值"""
        try:
            cfg = get_xianxia_config()
            self.realm_order = cfg.rank.get("levels", [])
            if not self.realm_order:
                self.realm_order = ["炼气", "筑基", "金丹", "元婴", "化神", "炼虚", "合体", "大乘", "渡劫"]
        except Exception:
            self.realm_order = ["炼气", "筑基", "金丹", "元婴", "化神", "炼虚", "合体", "大乘", "渡劫"]

    def validate(
        self,
        event_dict: Dict[str, Any],
        predicates: Dict[str, Predicate],
        world_state: Optional[Any] = None
    ) -> Dict[str, Any]:
        event_type = event_dict.get('type')
        if not event_type:
            return {"passed": True, "severity": "info", "missing_preconditions": [], "suggestions": []}

        # ===== 特殊处理：realm_upgrade（境界突破）=====
        if event_type == 'realm_upgrade':
            actor = event_dict.get('actor')
            to_major_realm = event_dict.get('to_major_realm')
            if actor and to_major_realm:
                # 从 predicates 中查找当前大境界
                current_realm = None
                for pred in predicates.values():
                    if pred.subject == actor and pred.relation == 'realm':
                        current_realm = pred.object
                        break

                # 情况1：没有当前境界（首次突破）→ 只能突破到最低级
                if current_realm is None:
                    lowest_realm = self.realm_order[0] if self.realm_order else "炼气"
                    if to_major_realm == lowest_realm:
                        logger.debug(f"First realm upgrade for {actor} to {lowest_realm}, allowing")
                        return {"passed": True, "severity": "info", "missing_preconditions": [], "suggestions": []}
                    else:
                        logger.warning(f"First realm upgrade for {actor} to {to_major_realm} not allowed (must be {lowest_realm})")
                        return {
                            "passed": False,
                            "severity": "error",
                            "missing_preconditions": [],
                            "suggestions": [f"首次突破只能达到{lowest_realm}期"],
                            "error_details": {
                                "type": "realm_upgrade_violation",
                                "current_realm": None,
                                "expected_realm": lowest_realm,
                                "actual_realm": to_major_realm,
                                "message": f"首次突破只能达到{lowest_realm}期，不能直接到{to_major_realm}"
                            }
                        }

                # 情况2：同大境界内层级提升（to_major_realm 与 current_realm 相同）
                if current_realm == to_major_realm:
                    logger.debug(f"Realm upgrade within same major realm ({current_realm} → {to_major_realm}), skipping rule check")
                    return {"passed": True, "severity": "info", "missing_preconditions": [], "suggestions": []}

                # 情况3：跨大境界突破 → 检查顺序（只能提升一级）
                try:
                    current_idx = self.realm_order.index(current_realm)
                    target_idx = self.realm_order.index(to_major_realm)
                    if target_idx == current_idx + 1:
                        logger.debug(f"Valid realm upgrade from {current_realm} to {to_major_realm}")
                        return {"passed": True, "severity": "info", "missing_preconditions": [], "suggestions": []}
                    else:
                        next_realm = self.realm_order[current_idx + 1] if current_idx + 1 < len(self.realm_order) else None
                        logger.warning(f"Invalid realm upgrade: {current_realm} → {to_major_realm} (must be sequential)")
                        return {
                            "passed": False,
                            "severity": "error",
                            "missing_preconditions": [],
                            "suggestions": [f"境界必须逐级突破，当前{current_realm}只能突破到{next_realm}，不能直接到{to_major_realm}"],
                            "error_details": {
                                "type": "realm_upgrade_violation",
                                "current_realm": current_realm,
                                "expected_realm": next_realm,
                                "actual_realm": to_major_realm,
                                "message": f"境界必须逐级突破，当前{current_realm}只能突破到{next_realm}，不能直接到{to_major_realm}"
                            }
                        }
                except ValueError as e:
                    logger.error(f"Unknown realm in order list: {current_realm} or {to_major_realm}: {e}")
                    return {
                        "passed": True,
                        "severity": "warning",
                        "missing_preconditions": [],
                        "suggestions": [f"未知境界 {current_realm} 或 {to_major_realm}，已跳过顺序检查"],
                        "error_details": {
                            "type": "unknown_realm",
                            "current_realm": current_realm,
                            "actual_realm": to_major_realm,
                            "message": f"未知境界 {current_realm} 或 {to_major_realm}"
                        }
                    }

        # ===== 正常规则检查（其他事件类型）=====
        rules = self.rule_engine.get_rules_for_event(event_type)
        if not rules:
            return {"passed": True, "severity": "info", "missing_preconditions": [], "suggestions": []}

        all_missing = []
        suggestions = []
        worst_severity = "info"
        for rule in rules:
            matched, bindings, missing = self.rule_engine.match_rule_with_event(rule, predicates, event_dict)
            if not matched:
                all_missing.extend(missing)
                suggestion_text = rule.suggestion
                for k, v in bindings.items():
                    suggestion_text = suggestion_text.replace(f"{{{k}}}", str(v))
                suggestions.append(suggestion_text)
                if rule.severity == "error":
                    worst_severity = "error"
                elif rule.severity == "warning" and worst_severity != "error":
                    worst_severity = "warning"
        passed = (worst_severity != "error")

        # ===== 约束检查（如果提供了 world_state 且包含 constraints）=====
        if world_state is not None and hasattr(world_state, 'constraints') and world_state.constraints:
            try:
                registry = ConstraintRegistry(world_state.constraints)
                actor = event_dict.get("actor") or event_dict.get("name")
                target = event_dict.get("target") or event_dict.get("to_char")
                violated = registry.check_violation(event_type, actor, target)
                if violated:
                    return {
                        "passed": False,
                        "severity": "error",
                        "missing_preconditions": [],
                        "suggestions": [f"违反约束: {v.description}" for v in violated]
                    }
            except Exception as e:
                logger.warning(f"Constraint check failed: {e}")

        return {
            "passed": passed,
            "severity": worst_severity,
            "missing_preconditions": all_missing,
            "suggestions": suggestions
        }

    def _render_suggestion(self, template: str, bindings: Dict, missing: List) -> str:
        text = template
        for key, value in bindings.items():
            text = text.replace(f"{{{key}}}", str(value))
        for m in missing:
            if isinstance(m.get('subject'), str) and m['subject'].startswith('?'):
                text = text.replace(f"{{{m['subject'][1:]}}}", "???")
        return text