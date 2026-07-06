# experiments/phase1/metrics.py
import re
import json
from typing import Dict, Any, List, Optional
# 不再导入 PlanningDirective，我们直接用字典
# from src.writing.planning_directive import PlanningDirective, PlanningRepresentation

def compute_surface_compliance(directive_dict: Dict, text: str) -> float:
    rep = directive_dict.get("representation", "summary")
    content = directive_dict.get("content", {})
    key_phrases = []
    if rep == "summary":
        summary = content.get("summary", "")
        words = re.findall(r'[\u4e00-\u9fff]{2,4}', summary)
        key_phrases = list(set(words))
    elif rep == "beat":
        key_phrases = content.get("beats", [])
    elif rep == "action":
        key_phrases = content.get("actions", [])
    elif rep == "intent":
        purpose = content.get("purpose", "")
        success = content.get("success_condition", "")
        sub = content.get("sub_goals", [])
        all_text = purpose + " " + success + " " + " ".join(sub)
        words = re.findall(r'[\u4e00-\u9fff]{2,4}', all_text)
        key_phrases = list(set(words))
    elif rep == "constraint":
        must = content.get("must_happen", [])
        all_text = " ".join(must)
        words = re.findall(r'[\u4e00-\u9fff]{2,4}', all_text)
        key_phrases = list(set(words))
    if not key_phrases:
        return 1.0
    text_clean = re.sub(r'[，。！？；：""“”‘’\n\t]', '', text)
    matched = sum(1 for p in key_phrases if p in text_clean)
    return matched / len(key_phrases)

def compute_deep_compliance(directive_dict: Dict, initial_state: Dict, final_state: Dict) -> float:
    expected = directive_dict.get("expected_outcome")
    if not expected:
        return 1.0
    # 简单比较：提取状态变化
    def extract_sig(state):
        sig = set()
        chars = state.get("characters", {})
        for name, info in chars.items():
            sig.add(f"realm:{name}:{info.get('realm','')}{info.get('realm_level',0)}")
        items = state.get("items", {})
        for item, info in items.items():
            sig.add(f"item:{item}:{info.get('owner','none')}")
        rels = state.get("relationships", {})
        for rel, val in rels.items():
            if abs(val) > 10:
                sig.add(f"rel:{rel}:{val}")
        flags = state.get("global_flags", {})
        for flag, val in flags.items():
            if val is True:
                sig.add(f"flag:{flag}:True")
        return sig
    init_sig = extract_sig(initial_state)
    final_sig = extract_sig(final_state)
    actual_changes = final_sig - init_sig
    expected_changes = set()
    if "events" in expected:
        for evt in expected["events"]:
            expected_changes.add(json.dumps(evt, ensure_ascii=False))
    else:
        for k, v in expected.items():
            expected_changes.add(f"{k}:{v}")
    if not expected_changes:
        return 1.0
    inter = len(actual_changes & expected_changes)
    union = len(actual_changes | expected_changes)
    return inter / union if union > 0 else 0.0

def compute_predictability(expected_projection: Dict, actual_projection: Dict) -> float:
    if not expected_projection or not actual_projection:
        return 0.5
    fields = ["focus", "loop", "attention", "question"]
    matches = sum(1 for f in fields if expected_projection.get(f) == actual_projection.get(f))
    return matches / len(fields)

def compute_rewrite_rate(validation_result: Optional[Dict]) -> float:
    if not validation_result:
        return 0.0
    return 0.0 if validation_result.get("passed", True) else 1.0

def compute_all_metrics(
    directive_dict: Dict,
    text: str,
    initial_state: Dict,
    final_state: Dict,
    validation_result: Optional[Dict],
    expected_projection: Optional[Dict] = None,
    actual_projection: Optional[Dict] = None,
) -> Dict[str, float]:
    return {
        "surface_compliance": compute_surface_compliance(directive_dict, text),
        "deep_compliance": compute_deep_compliance(directive_dict, initial_state, final_state),
        "predictability": compute_predictability(expected_projection, actual_projection),
        "rewrite_rate": compute_rewrite_rate(validation_result),
    }