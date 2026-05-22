"""规则引擎 - 加载YAML规则，匹配 precondition，支持变量绑定"""
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from .predicate import Predicate, normalize_object


@dataclass
class Precondition:
    subject: str
    relation: str
    object: Any
    negated: bool = False
    min_confidence: float = 0.0
    allowed_source_types: List[str] = field(default_factory=list)


@dataclass
class Rule:
    id: str
    precondition: List[Precondition]
    trigger_event_type: Optional[str]
    severity: str
    suggestion: str
    enables: List[str] = field(default_factory=list)
    cooldown: int = 0
    hint: str = ""                              

class RuleEngine:
    def __init__(self, rules_path: Optional[str] = None):
        if rules_path is None:
            rules_path = Path(__file__).parent / "rules" / "causality_rules.yaml"
        self.rules = self._load_rules(rules_path)
        self._index_by_event_type = self._build_index()

    def _load_rules(self, path: Path) -> List[Rule]:
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        rules = []
        for r in data['rules']:
            preconditions = []
            for p in r['precondition']:
                preconditions.append(Precondition(
                    subject=p['subject'],
                    relation=p['relation'],
                    object=p['object'],
                    negated=p.get('negated', False),
                    min_confidence=p.get('min_confidence', 0.0),
                    allowed_source_types=p.get('allowed_source_types', [])
                ))
            rules.append(Rule(
                id=r['id'],
                precondition=preconditions,
                trigger_event_type=r.get('trigger_event_type'),
                severity=r.get('severity', 'info'),
                suggestion=r.get('suggestion', ''),
                enables=r.get('enables', []),
                cooldown=r.get('cooldown', 0),
                hint=r.get('hint', '')
            ))                       
        return rules

    def _build_index(self) -> Dict[str, List[Rule]]:
        index = {}
        for rule in self.rules:
            if rule.trigger_event_type:
                index.setdefault(rule.trigger_event_type, []).append(rule)
        return index

    def get_rules_for_event(self, event_type: str) -> List[Rule]:
        return self._index_by_event_type.get(event_type, [])

    def match_rule_with_event(
        self,
        rule: Rule,
        predicates: Dict[str, Predicate],
        event: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any], List[Dict[str, Any]]]:
        from .predicate import normalize_object

        # 从事件中提取初始绑定（规范化）
        bindings = {}
        for key in ('actor', 'item', 'target', 'location', 'from_char', 'to_char', 'realm'):
            if key in event:
                bindings[key] = normalize_object(event[key])

        for prec in rule.precondition:
            # 构建期望的 identity_key
            # 确定主语
            if prec.subject.startswith('?'):
                var = prec.subject[1:]
                if var not in bindings:
                    return False, bindings, [{"subject": prec.subject, "relation": prec.relation, "object": prec.object, "negated": prec.negated}]
                subj_val = bindings[var]
            else:
                subj_val = normalize_object(prec.subject)

            # 确定对象
            if isinstance(prec.object, str) and prec.object.startswith('?'):
                var = prec.object[1:]
                if var not in bindings:
                    return False, bindings, [{"subject": prec.subject, "relation": prec.relation, "object": prec.object, "negated": prec.negated}]
                obj_val = bindings[var]
            else:
                # 常量对象（布尔值或普通值）
                if isinstance(prec.object, bool):
                    obj_val = prec.object
                else:
                    obj_val = normalize_object(prec.object)

            # 构建预期的 identity_key
            neg = "not_" if prec.negated else ""
            expected_key = f"{subj_val}|{prec.relation}|{neg}{obj_val}"
            
            # 在 predicates 中查找
            pred = predicates.get(expected_key)
            if pred is None:
                return False, bindings, [{"subject": prec.subject, "relation": prec.relation, "object": prec.object, "negated": prec.negated}]

            # 额外检查置信度和来源类型
            if pred.confidence < prec.min_confidence:
                return False, bindings, [{"subject": prec.subject, "relation": prec.relation, "object": prec.object, "negated": prec.negated}]
            if prec.allowed_source_types and pred.source_event_type not in prec.allowed_source_types:
                return False, bindings, [{"subject": prec.subject, "relation": prec.relation, "object": prec.object, "negated": prec.negated}]

        return True, bindings, []    