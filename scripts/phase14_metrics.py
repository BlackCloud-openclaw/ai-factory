#!/usr/bin/env python3
"""Phase 14.0C-3: Validator Metrics Calculator"""

import json
import sys
from collections import Counter, defaultdict
from typing import List, Dict, Any

def extract_validator_outputs(log_path: str) -> List[Dict]:
    outputs = []
    with open(log_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                # 尝试多个可能的路径
                val_out = None
                if 'validator_output' in data:
                    val_out = data['validator_output']
                elif 'extra' in data and 'validator_output' in data['extra']:
                    val_out = data['extra']['validator_output']
                elif 'result' in data and 'validator_output' in data['result']:
                    val_out = data['result']['validator_output']
                if val_out:
                    outputs.append(val_out)
            except:
                continue
    return outputs

def compute_metrics(outputs: List[Dict]) -> Dict:
    total = len(outputs)
    if total == 0:
        return {"error": "No ValidatorOutput found"}

    statuses = Counter()
    violations = []
    confidences = []
    rules = Counter()

    for out in outputs:
        statuses[out.get('status', 'unknown')] += 1
        confidences.append(out.get('confidence', 0.0))
        for v in out.get('violations', []):
            violations.append(v)
            rules[v.get('rule_id', 'unknown')] += 1

    passed = statuses.get('passed', 0)
    degraded = statuses.get('degraded', 0)
    failed = statuses.get('failed', 0)

    # 假设 retry 信息包含在 metadata 或其他字段，暂未提取
    # 暂时返回基础统计
    return {
        "total_scenes": total,
        "status_distribution": dict(statuses),
        "pass_rate": passed / total if total else 0,
        "degraded_rate": degraded / total if total else 0,
        "failure_rate": failed / total if total else 0,
        "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
        "total_violations": len(violations),
        "avg_violations_per_scene": len(violations) / total if total else 0,
        "top_rules": rules.most_common(5),
    }

if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else "logs/ai_factory.log"
    outputs = extract_validator_outputs(log_path)
    metrics = compute_metrics(outputs)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))