# src/writing/contract_normalizer.py
"""
Contract Normalizer - Phase 13.2.3A + Phase 14.0A-2
"""

import hashlib
import json
import logging
import re
from copy import deepcopy
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
from src.writing.state_change_types import StateChangeType
from src.writing.planning_contract import (
    PlanningContract,
    StateChange,
    Observables,
    ContractEnrichment,
    SignalSource,
    Constraint,
    ExecutionUnit,
)

logger = logging.getLogger(__name__)


@dataclass
class StateChangeRule:
    keywords: Set[str]
    state_type: StateChangeType
    confidence: float
    description: str = ""


INFERENCE_RULES = [
    StateChangeRule(
        keywords={"发现", "揭示", "确认", "验证", "证明", "揭晓"},
        state_type=StateChangeType.KNOWLEDGE_GAIN,
        confidence=0.95,
        description="知识获取",
    ),
    StateChangeRule(
        keywords={"获得", "捡到", "夺取", "缴获", "拾取", "拿到"},
        state_type=StateChangeType.INVENTORY_ACQUIRE,
        confidence=0.95,
        description="物品获得",
    ),
    StateChangeRule(
        keywords={"进入", "抵达", "来到", "返回", "离开", "走出"},
        state_type=StateChangeType.LOCATION_CHANGE,
        confidence=0.90,
        description="位置变化",
    ),
    StateChangeRule(
        keywords={"突破", "晋升", "晋级", "渡劫", "破境"},
        state_type=StateChangeType.REALM_CHANGE,
        confidence=0.90,
        description="境界提升",
    ),
    StateChangeRule(
        keywords={"结盟", "交恶", "和解", "决裂", "结仇"},
        state_type=StateChangeType.RELATIONSHIP_CHANGE,
        confidence=0.85,
        description="关系变化",
    ),
    StateChangeRule(
        keywords={"触发", "激活", "启动", "破解", "解除", "开启"},
        state_type=StateChangeType.PLOT_FLAG,
        confidence=0.90,
        description="剧情标记",
    ),
    StateChangeRule(
        keywords={"达成", "签署", "同意", "承诺"},
        state_type=StateChangeType.RELATIONSHIP_CHANGE,
        confidence=0.80,
        description="协议达成",
    ),
]


class ContractNormalizer:
    def __init__(self):
        self._audit_logs = []

    def normalize(self, contract: PlanningContract) -> PlanningContract:
        current_hash = self._compute_input_hash(contract)
        if (
            contract.enrichment
            and contract.enrichment.enriched
            and contract.enrichment.input_hash == current_hash
        ):
            logger.debug(f"Contract {contract.scene_id} unchanged, skipping normalization.")
            return contract

        normalized = deepcopy(contract)
        if normalized.enrichment and normalized.enrichment.enriched:
            normalized.enrichment = ContractEnrichment()

        if not normalized.observables:
            normalized.observables = Observables()

        existing_ids = {sc.id for sc in normalized.observables.state_changes if hasattr(sc, 'id')}
        inferred_changes = self._infer_from_must_events(normalized, existing_ids)

        rules_applied = []
        sources = {}

        if inferred_changes:
            normalized.observables.state_changes.extend(inferred_changes)
            rules_applied.append("INFER_STATE_CHANGES_FROM_EVENTS")

        if not normalized.observables.state_changes:
            fallback_changes = self._fallback_infer(
                normalized.execution.units,
                normalized.scene_id,
                existing_ids
            )
            if fallback_changes:
                normalized.observables.state_changes.extend(fallback_changes)
                rules_applied.append("INFER_STATE_CHANGES_FALLBACK")

        for sc in normalized.observables.state_changes:
            if hasattr(sc, 'id'):
                if sc.id not in sources:
                    is_llm = any(
                        hasattr(orig, 'id') and orig.id == sc.id
                        for orig in contract.observables.state_changes
                    )
                    sources[sc.id] = SignalSource.LLM if is_llm else SignalSource.INFERRED

        if not normalized.constraints:
            inferred_constraints = self._infer_constraints(normalized.execution.units)
            if inferred_constraints:
                normalized.constraints = inferred_constraints
                rules_applied.append("INFER_CONSTRAINTS")
                for c in normalized.constraints:
                    if hasattr(c, 'id'):
                        sources[c.id] = SignalSource.INFERRED

        if rules_applied:
            normalized.enrichment.mark_enriched(current_hash)
            normalized.enrichment.rules_applied = rules_applied
            normalized.enrichment.sources = {
                k: v.value if hasattr(v, 'value') else str(v)
                for k, v in sources.items()
            }
            self._record_audit(normalized)

        return normalized

    def _extract_must_events(self, contract: PlanningContract) -> List[str]:
        """从 execution.units.description 提取 must_events"""
        events = []
        if contract.execution and contract.execution.units:
            for unit in contract.execution.units:
                desc = getattr(unit, 'description', '') or ''
                if desc:
                    events.append(desc)
        return events

    def _infer_from_must_events(self, contract: PlanningContract, existing_ids: set) -> List[StateChange]:
        must_events = self._extract_must_events(contract)
        if not must_events:
            return []

        inferred = []
        for event_text in must_events:
            if "推进" in event_text and "剧情" in event_text:
                continue

            for rule in INFERENCE_RULES:
                if any(kw in event_text for kw in rule.keywords):
                    sc = self._create_state_change(rule, event_text, contract.scene_id)
                    if sc and sc.id not in existing_ids:
                        existing_ids.add(sc.id)
                        inferred.append(sc)
        return inferred

    def _create_state_change(self, rule: StateChangeRule, event_text: str, contract_id: str) -> Optional[StateChange]:
        """
        根据规则创建 StateChange。

        核心原则：
        - 不填充默认事实（actor=None, to_major_realm=None, delta=None, location=None）
        - 只记录推断出的信息
        - 标记 source=SignalSource.INFERRED 和 confidence
        """
        import hashlib
        import re

        # 生成稳定 ID
        clean = re.sub(r'[\s，。、！？；：""''（）]', '', event_text)
        raw = f"{contract_id}|{rule.state_type.value}|{clean}"
        sc_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]

        # 提取目标（启发式，不保证准确，但用于标识）
        target = ""

        # ====================================================================
        # 1. 关系变化：提取 "与X" 中的 X
        # ====================================================================
        if rule.state_type == StateChangeType.RELATIONSHIP_CHANGE:
            match = re.search(r'与([\u4e00-\u9fff]{2,4})', event_text)
            if match:
                target = match.group(1)
                # 如果 target 末尾是关系动词首字，截断（如"交恶"中的"交"）
                relation_endings = {"交", "盟", "解", "裂", "仇"}
                if target and target[-1] in relation_endings:
                    target = target[:-1]
            else:
                # fallback: 取事件中最后一个 2-4 字词（排除动作词）
                words = re.findall(r'[\u4e00-\u9fff]{2,4}', event_text)
                if words:
                    for w in reversed(words):
                        if not any(kw in w for kw in rule.keywords):
                            target = w
                            break

        # ====================================================================
        # 2. 知识获取 / 物品获得 / 位置变化 / 剧情标记：提取动作后的名词短语
        # ====================================================================
        elif rule.state_type in (
            StateChangeType.KNOWLEDGE_GAIN,
            StateChangeType.INVENTORY_ACQUIRE,
            StateChangeType.LOCATION_CHANGE,
            StateChangeType.PLOT_FLAG
        ):
            # 尝试匹配 "动作词 + 名词短语"
            for action in rule.keywords:
                if action in event_text:
                    idx = event_text.find(action) + len(action)
                    rest = event_text[idx:].strip()
                    if rest:
                        match = re.search(r'^([\u4e00-\u9fff]{2,10})', rest)
                        if match:
                            target = match.group(1)
                            break
            # fallback: 取最后一个 2-4 字词
            if not target:
                nouns = re.findall(r'[\u4e00-\u9fff]{2,4}', event_text)
                if nouns:
                    target = nouns[-1]

        # ====================================================================
        # 3. 境界提升：特殊处理，不提取目标（因为通常无具体对象）
        # ====================================================================
        elif rule.state_type == StateChangeType.REALM_CHANGE:
            # 境界提升不提取 target，保持 None
            pass

        # ====================================================================
        # 4. 通用 fallback
        # ====================================================================
        else:
            nouns = re.findall(r'[\u4e00-\u9fff]{2,4}', event_text)
            if nouns:
                target = nouns[-1]

        # ====================================================================
        # 创建 StateChange（不填充默认事实）
        # ====================================================================
        # 注意：source 使用 SignalSource.INFERRED 枚举
        if rule.state_type == StateChangeType.KNOWLEDGE_GAIN:
            return StateChange(
                id=sc_id,
                type=rule.state_type.value,
                source=SignalSource.INFERRED,   # ✅ 枚举
                confidence=rule.confidence,
                name=f"knowledge_{target}" if target else "knowledge_gain",
                value=True,
            )
        elif rule.state_type == StateChangeType.INVENTORY_ACQUIRE:
            return StateChange(
                id=sc_id,
                type=rule.state_type.value,
                source=SignalSource.INFERRED,
                confidence=rule.confidence,
                actor=None,
                item=target or "item",
                operation="acquire",
                quantity=1,
            )
        elif rule.state_type == StateChangeType.LOCATION_CHANGE:
            return StateChange(
                id=sc_id,
                type=rule.state_type.value,
                source=SignalSource.INFERRED,
                confidence=rule.confidence,
                actor=None,
                location=target or None,
            )
        elif rule.state_type == StateChangeType.REALM_CHANGE:
            return StateChange(
                id=sc_id,
                type=rule.state_type.value,
                source=SignalSource.INFERRED,
                confidence=rule.confidence,
                actor=None,
                to_major_realm=None,
                to_minor_stage=None,
            )
        elif rule.state_type == StateChangeType.RELATIONSHIP_CHANGE:
            return StateChange(
                id=sc_id,
                type=rule.state_type.value,
                source=SignalSource.INFERRED,
                confidence=rule.confidence,
                from_char=None,
                to_char=target or "other",
                delta=None,
            )
        elif rule.state_type == StateChangeType.PLOT_FLAG:
            return StateChange(
                id=sc_id,
                type=rule.state_type.value,
                source=SignalSource.INFERRED,
                confidence=rule.confidence,
                name=target or "flag_triggered",
                value=True,
            )
        return None

    def _fallback_infer(self, units, contract_id, existing_ids):
        from src.writing.event_classifier import EventClassifier
        from src.writing.state_change_factory import StateChangeFactory
        classifier = EventClassifier()
        factory = StateChangeFactory()
        inferred = []
        for unit in units:
            text = getattr(unit, 'description', '') or str(unit)
            if len(text) < 5 or "推进主线" in text:
                continue
            event_types = classifier.classify(text)
            if not event_types:
                continue
            context = {"text": text}
            match = re.match(r'^([\u4e00-\u9fff]{2,4})', text)
            if match:
                context["actor"] = match.group(1)
            for event_type in event_types:
                sc = factory.create(event_type, context, contract_id)
                if sc and sc.id not in existing_ids:
                    existing_ids.add(sc.id)
                    inferred.append(sc)
        return inferred

    def _infer_constraints(self, units):
        from src.writing.planning_contract import Constraint
        constraints = []
        for unit in units:
            text = getattr(unit, 'description', '') or str(unit)
            if "禁止" in text or "不得" in text:
                target = text.replace("禁止", "").replace("不得", "").strip()
                if target:
                    import hashlib
                    cid = hashlib.sha256(f"inferred_constraint_{target}".encode()).hexdigest()[:8]
                    constraints.append(Constraint(
                        id=cid,
                        type="forbidden",
                        target=target,
                        condition=None,
                    ))
        return constraints

    def _compute_input_hash(self, contract):
        data = {
            "scene_id": contract.scene_id,
            "execution_units": [
                {"id": u.id, "description": u.description}
                for u in contract.execution.units
            ],
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def _record_audit(self, contract):
        audit_entry = {
            "contract_id": contract.scene_id,
            "normalizer_version": "14.0A-2.1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_hash": contract.enrichment.input_hash,
            "signals": {
                "state_changes": {
                    "total": len(contract.observables.state_changes),
                    "llm": sum(1 for sc in contract.observables.state_changes if sc.source == SignalSource.LLM),
                    "inferred": sum(1 for sc in contract.observables.state_changes if sc.source == SignalSource.INFERRED),
                }
            },
            "rules_applied": contract.enrichment.rules_applied,
            "enrichment_applied": contract.enrichment.enriched,
        }
        logger.info(f"Contract Normalizer Audit: {json.dumps(audit_entry, indent=2)}")
        self._audit_logs.append(audit_entry)
        return audit_entry

    def get_audit_logs(self):
        return self._audit_logs